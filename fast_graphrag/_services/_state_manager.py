import asyncio
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Awaitable, Iterable, List, Optional, Tuple, Type

import numpy as np
from scipy.sparse import csr_matrix

from fast_graphrag._llm import BaseLLMService
from fast_graphrag._storage._base import (
    BaseBlobStorage,
    BaseGraphStorage,
    BaseStorage,
)
from fast_graphrag._storage._blob_pickle import PickleBlobStorage
from fast_graphrag._storage._namespace import Workspace
from fast_graphrag._types import TChunk, TContext, TEmbedding, TEntity, THash, TId, TRelation
from fast_graphrag._utils import csr_from_indices_list, extract_sorted_scores, logger

from ._base import BaseStateManagerService


@dataclass
class DefaultStateManagerService(BaseStateManagerService[TEntity, TRelation, THash, TChunk, TId, TEmbedding]):
    blob_storage_cls: Type[BaseBlobStorage[csr_matrix]] = field(default=PickleBlobStorage)

    def __post_init__(self):
        self._workspace = Workspace.new(working_dir=self.working_dir)

        self.graph_storage.namespace = self._workspace.make_for("graph")
        self.entity_storage.namespace = self._workspace.make_for("entities")
        self.chunk_storage.namespace = self._workspace.make_for("chunks")

        self._entities_to_relationships: BaseBlobStorage[csr_matrix] = self.blob_storage_cls(
            namespace=self._workspace.make_for("map_e2r"), config=None
        )
        self._relationships_to_chunks: BaseBlobStorage[csr_matrix] = self.blob_storage_cls(
            namespace=self._workspace.make_for("map_r2c"), config=None
        )

    async def filter_new_chunks(self, chunks_per_data: Iterable[Iterable[TChunk]]) -> List[List[TChunk]]:
        flattened_chunks = [chunk for chunks in chunks_per_data for chunk in chunks]
        if len(flattened_chunks) == 0:
            return []

        new_chunks_mask = await self.chunk_storage.mask_new(keys=[c.id for c in flattened_chunks])

        i = iter(new_chunks_mask)
        new_chunks = [[chunk for chunk in chunks if next(i)] for chunks in chunks_per_data]

        return new_chunks

    async def upsert(
        self,
        llm: BaseLLMService,
        subgraphs: List[asyncio.Future[Optional[BaseGraphStorage[TEntity, TRelation, TId]]]],
        documents: Iterable[Iterable[TChunk]],
    ) -> None:
        nodes: Iterable[List[TEntity]]
        edges: Iterable[List[TRelation]]

        async def _get_graphs(
            fgraph: asyncio.Future[Optional[BaseGraphStorage[TEntity, TRelation, TId]]],
        ) -> Optional[Tuple[List[TEntity], List[TRelation]]]:
            graph = await fgraph
            if graph is None:
                return None

            nodes = [t for i in range(await graph.node_count()) if (t := await graph.get_node_by_index(i)) is not None]
            edges = [t for i in range(await graph.edge_count()) if (t := await graph.get_edge_by_index(i)) is not None]

            return (nodes, edges)

        graphs = await asyncio.gather(*[_get_graphs(fgraph) for fgraph in subgraphs])
        graphs = [graph for graph in graphs if graph is not None]
        if len(graphs) == 0:
            return

        nodes, edges = zip(*graphs)

        _, upserted_nodes = await self.node_upsert_policy(llm, self.graph_storage, chain(*nodes))
        _, _ = await self.edge_upsert_policy(llm, self.graph_storage, chain(*edges))

        # Insert entities in entity_storage
        embeddings = await self.embedding_service.get_embedding(texts=[d.to_str() for _, d in upserted_nodes])
        await self.entity_storage.upsert(ids=(i for i, _ in upserted_nodes), embeddings=embeddings)

        # Insert chunks in chunk_storage
        flattened_chunks = [chunk for chunks in documents for chunk in chunks]
        await self.chunk_storage.upsert(keys=[chunk.id for chunk in flattened_chunks], values=flattened_chunks)

    async def get_context(self, entities: Iterable[TEntity]) -> Optional[TContext[TEntity, TRelation, THash, TChunk]]:
        try:
            entity_names = [entity.name for entity in entities]
            if len(entity_names) == 0:
                return None

            query_embeddings = await self.embedding_service.get_embedding(entity_names)

            # Similarity-search over entities
            vdb_entity_scores = await self._score_entities_by_vectordb(query_embeddings=query_embeddings)

            if vdb_entity_scores.nnz == 0:
                return None
        except Exception as e:
            logger.error(f"Error during information extraction and scoring for query entities {entities}.\n{e}")
            raise e

        # Score entities
        try:
            graph_entity_scores = self.entity_ranking_policy(
                await self._score_entities_by_graph(entity_scores=vdb_entity_scores)
            )
        except Exception as e:
            logger.error(f"Error during graph scoring for entities. Non-zero elements: {vdb_entity_scores.nnz}.\n{e}")
            raise e

        try:
            # All score vectors should be row vectors
            indices, s_entity_scores = extract_sorted_scores(graph_entity_scores)
            s_entity_scores /= np.max(s_entity_scores)  # Normalise scores
            relevant_entities: List[TEntity] = []
            for i in indices:
                entity = await self.graph_storage.get_node_by_index(i)
                if entity is not None:
                    relevant_entities.append(entity)

            # Extract relevant relationships
            relation_scores = self.relation_ranking_policy(
                await self._score_relationships_by_entities(entity_scores=graph_entity_scores)
            )

            indices, s_relation_scores = extract_sorted_scores(relation_scores)
            s_relation_scores /= np.max(s_relation_scores)  # Normalise scores
            relevant_relationships: List[TRelation] = []
            for i in indices:
                relationship = await self.graph_storage.get_edge_by_index(i)
                if relationship is not None:
                    relevant_relationships.append(relationship)

            # Extract relevant chunks
            chunk_scores = self.chunk_ranking_policy(
                await self._score_chunks_by_relations(relationships_score=relation_scores)
            )
            indices, s_chunk_scores = extract_sorted_scores(chunk_scores)
            s_chunk_scores /= np.max(s_chunk_scores)  # Normalise scores
            relevant_chunks: List[TChunk] = []
            for chunk in await self.chunk_storage.get_by_index(indices):
                if chunk is not None:
                    relevant_chunks.append(chunk)

            return TContext(
                entities=relevant_entities,
                relationships=relevant_relationships,
                chunks=relevant_chunks,
                entity_scores=s_entity_scores.tolist(),
                relationship_scores=s_relation_scores.tolist(),
                chunk_scores=s_chunk_scores.tolist(),
            )
        except Exception as e:
            logger.error(f"Error during scoring of chunks and relationships.\n{e}")
            raise e

    async def _get_entities_to_num_docs(self) -> Any:
        raise NotImplementedError

    async def _score_entities_by_vectordb(self, query_embeddings: Iterable[TEmbedding], top_k: int = 1) -> csr_matrix:
        if top_k != 1:
            raise NotImplementedError("Top-k != 1 is not supported yet.")
        if self.node_specificity:
            raise NotImplementedError("Node specificity is not supported yet.")

        all_entity_probs_by_query_entity = await self.entity_storage.score_all(
            np.array(query_embeddings), top_k=top_k
        )  # (#query_entities, #all_entities)

        # TODO: if top_k > 1, we need to aggregate the scores here
        # At the moment, we just take the max and, since we normalise, the max value of a top_k=1 is always 1.0
        if all_entity_probs_by_query_entity.shape[1] == 0:
            return all_entity_probs_by_query_entity
        all_entity_weights: csr_matrix = all_entity_probs_by_query_entity.max(axis=0)  # (1, #all_entities)

        if self.node_specificity:
            all_entity_weights = all_entity_weights.multiply(1.0 / await self._get_entities_to_num_docs())

        return all_entity_weights

    async def _score_entities_by_graph(self, entity_scores: Optional[csr_matrix]) -> csr_matrix:
        graph_weighted_scores = await self.graph_storage.score_nodes(entity_scores)
        node_scores = csr_matrix(graph_weighted_scores)  # (1, #entities)
        return node_scores

    async def _score_relationships_by_entities(self, entity_scores: csr_matrix) -> csr_matrix:
        e2r = await self._entities_to_relationships.get()
        if e2r is None:
            logger.warning("No entities to relationships map was loaded.")
            return csr_matrix((1, await self.graph_storage.edge_count()))

        return entity_scores.dot(e2r)  # (1, #entities) x (#entities, #relationships) => (1, #relationships)

    async def _score_chunks_by_relations(self, relationships_score: csr_matrix) -> csr_matrix:
        c2r = await self._relationships_to_chunks.get()
        if c2r is None:
            logger.warning("No relationships to chunks map was loaded.")
            return csr_matrix((1, await self.chunk_storage.size()))
        return relationships_score.dot(c2r)  # (1, #relationships) x (#relationships, #chunks) => (1, #chunks)

    ####################################################################################################

    # I/O management
    ####################################################################################################

    async def query_start(self):
        storages: List[BaseStorage] = [
            self.graph_storage,
            self.entity_storage,
            self.chunk_storage,
            self._relationships_to_chunks,
            self._entities_to_relationships,
        ]

        def _fn():
            tasks: List[Awaitable[Any]] = []
            for storage_inst in storages:
                tasks.append(storage_inst.query_start())
            return asyncio.gather(*tasks)

        await self._workspace.with_checkpoints(_fn)

        for storage_inst in storages:
            storage_inst.set_in_progress(True)

    async def query_done(self):
        tasks: List[Awaitable[Any]] = []
        storages: List[BaseStorage] = [
            self.graph_storage,
            self.entity_storage,
            self.chunk_storage,
            self._relationships_to_chunks,
            self._entities_to_relationships,
        ]
        for storage_inst in storages:
            tasks.append(storage_inst.query_done())
        await asyncio.gather(*tasks)

        for storage_inst in storages:
            storage_inst.set_in_progress(False)

    async def insert_start(self):
        storages: List[BaseStorage] = [
            self.graph_storage,
            self.entity_storage,
            self.chunk_storage,
            self._relationships_to_chunks,
            self._entities_to_relationships,
        ]

        def _fn():
            tasks: List[Awaitable[Any]] = []
            for storage_inst in storages:
                tasks.append(storage_inst.insert_start())
            return asyncio.gather(*tasks)

        await self._workspace.with_checkpoints(_fn)

        for storage_inst in storages:
            storage_inst.set_in_progress(True)

    async def insert_done(self):
        await self._entities_to_relationships.set(await self.graph_storage.get_entities_to_relationships_map())

        raw_relationships_to_chunks = await self.graph_storage.get_relationships_attrs(key="chunks")
        # Map Chunk IDs to indices
        raw_relationships_to_chunks = [
            [i for i in await self.chunk_storage.get_index(chunk_ids) if i is not None]
            for chunk_ids in raw_relationships_to_chunks
        ]
        await self._relationships_to_chunks.set(
            csr_from_indices_list(
                raw_relationships_to_chunks, shape=(len(raw_relationships_to_chunks), await self.chunk_storage.size())
            )
        )

        tasks: List[Awaitable[Any]] = []
        storages: List[BaseStorage] = [
            self.graph_storage,
            self.entity_storage,
            self.chunk_storage,
            self._relationships_to_chunks,
            self._entities_to_relationships,
        ]
        for storage_inst in storages:
            tasks.append(storage_inst.insert_done())
        await asyncio.gather(*tasks)

        for storage_inst in storages:
            storage_inst.set_in_progress(False)
