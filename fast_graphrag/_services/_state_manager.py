import asyncio
from dataclasses import dataclass, field
from itertools import chain
import pickle
from typing import (
    Any,
    Awaitable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    cast,
    Union,
)
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix, vstack
from tqdm import tqdm

from fast_graphrag._llm import BaseLLMService
from fast_graphrag._storage._base import (
    BaseBlobStorage,
    BaseGraphStorage,
    BaseStorage,
)
from fast_graphrag._storage._blob_pickle import PickleBlobStorage
from fast_graphrag._storage._namespace import Workspace
from fast_graphrag._types import (
    TChunk,
    TContext,
    TEmbedding,
    TEntity,
    THash,
    TId,
    TIndex,
    TRelation,
    TScore,
)
from fast_graphrag._utils import csr_from_indices_list, extract_sorted_scores, logger

from ._base import BaseStateManagerService


@dataclass
class DefaultStateManagerService(BaseStateManagerService[TEntity, TRelation, THash, TChunk, TId, TEmbedding]):
    blob_storage_cls: Type[BaseBlobStorage[csr_matrix]] = field(default=PickleBlobStorage)
    insert_similarity_score_threshold: float = field(default=0.9)
    query_similarity_score_threshold: Optional[float] = field(default=0.7)

    def __post_init__(self):
        assert self.workspace is not None, "Workspace must be provided."

        self.graph_storage.namespace = self.workspace.make_for("graph")
        self.entity_storage.namespace = self.workspace.make_for("entities")
        self.chunk_storage.namespace = self.workspace.make_for("chunks")

        self._entities_to_relationships: BaseBlobStorage[csr_matrix] = self.blob_storage_cls(
            namespace=self.workspace.make_for("map_e2r"), config=None
        )
        self._relationships_to_chunks: BaseBlobStorage[csr_matrix] = self.blob_storage_cls(
            namespace=self.workspace.make_for("map_r2c"), config=None
        )

    async def get_num_entities(self) -> int:
        return await self.graph_storage.node_count()

    async def get_num_relations(self) -> int:
        return await self.graph_storage.edge_count()

    async def get_num_chunks(self) -> int:
        return await self.chunk_storage.size()

    async def filter_new_chunks(self, chunks_per_data: Iterable[Iterable[TChunk]]) -> List[List[TChunk]]:
        flattened_chunks = [chunk for chunks in chunks_per_data for chunk in chunks]
        if len(flattened_chunks) == 0:
            return []

        new_chunks_mask = await self.chunk_storage.mask_new(keys=[c.id for c in flattened_chunks])

        i = iter(new_chunks_mask)
        new_chunks = [[chunk for chunk in chunks if next(i)] for chunks in chunks_per_data]

        return new_chunks

    async def batch_prepare_node_summaries(
        self,
        graphs,
        summarize_node_prompt_path: Path,
    ):
        nodes, edges = zip(*graphs)
        await self.batch_node_upsert_prompt_policy(
            self.graph_storage,
            chain(*nodes),
            summarize_node_prompt_path,
        )

    async def batch_prepare_edge_summaries(
        self,
        graphs,
        summarize_edge_prompt_path: Path,
    ):
        nodes, edges = zip(*graphs)
        await self.batch_edge_upsert_prompt_policy(
            self.graph_storage,
            chain(*edges),
            summarize_edge_prompt_path,
        )

    async def batch_upsert_nodes(
        self,
        llm: BaseLLMService,
        graphs,
        summarize_node_output: dict,
        show_progress: bool = True,
    ):
        nodes: Iterable[List[TEntity]]
        edges: Iterable[List[TRelation]]

        nodes, edges = zip(*graphs)

        _, upserted_nodes = await self.batch_node_upsert_policy(
            llm,
            self.graph_storage,
            chain(*nodes),
            summarize_node_output,
        )
        return upserted_nodes

    async def batch_upsert(
        self,
        llm: BaseLLMService,
        graphs,
        documents: Iterable[Iterable[TChunk]],
        upserted_nodes,
        summarize_edge_output: dict,
        show_progress: bool = True,
    ) -> Union[bool, None]:
        nodes: Iterable[List[TEntity]]
        edges: Iterable[List[TRelation]]

        progress_bar = tqdm(total=5, disable=not show_progress, desc="Building...")
        nodes, edges = zip(*graphs)

        progress_bar.set_description("Building... [upserting edges]")

        _, _ = await self.batch_edge_upsert_policy(
            llm,
            self.graph_storage,
            chain(*edges),
            summarize_edge_output,
        )
        progress_bar.update(1)

        # STEP (2): Computing entity embeddings
        progress_bar.set_description("Building... [computing embeddings]")
        # Insert entities in entity_storage
        # TODO make sure TEntity never has none values
        # TODO make this bit batch async too
        embeddings = await self.embedding_service.encode(texts=[d.to_str() for _, d in upserted_nodes if d is not None])
        progress_bar.update(1)
        await self.entity_storage.upsert(ids=(i for i, _ in upserted_nodes), embeddings=embeddings)
        progress_bar.update(1)

        # STEP: Entity deduplication
        # Note that get_knn will very likely return the same entity as the most similar one, so we remove it
        # when selecting the index order.
        progress_bar.set_description("Building... [entity deduplication]")
        upserted_indices = np.array([i for i, _ in upserted_nodes]).reshape(-1, 1)
        similar_indices, scores = await self.entity_storage.get_knn(embeddings, top_k=3)
        similar_indices = np.array(similar_indices)
        scores = np.array(scores)

        # Create matrix with the indices with a score higher than the threshold
        # We use the fact that similarity scores are symmetric between entity pairs,
        # so we only select half of that by index order
        similar_indices[
            (scores < self.insert_similarity_score_threshold)
            | (similar_indices <= upserted_indices)  # remove indices smaller or equal the entity
        ] = 0  # 0 can be used here (not 100% sure, but 99% sure)
        progress_bar.update(1)

        # | entity_index  | similar_indices[] |
        # |---------------|------------------------|
        # | 1             | 0, 7, 12, 0, 9         |

        # STEP: insert identity edges
        progress_bar.set_description("Building... [identity edges]")

        async def _insert_identiy_edges(
            source_index: TIndex, target_indices: npt.NDArray[np.int32]
        ) -> Iterable[Tuple[TIndex, TIndex]]:
            return [
                (source_index, idx)
                for idx in target_indices
                if idx != 0 and not await self.graph_storage.are_neighbours(source_index, idx)
            ]

        new_edge_indices = list(
            chain(
                *await asyncio.gather(*[_insert_identiy_edges(i, indices) for i, indices in enumerate(similar_indices)])
            )
        )
        new_edges_attrs: Dict[str, Any] = {
            "description": ["is"] * len(new_edge_indices),
            "chunks": [[]] * len(new_edge_indices),
        }
        await self.graph_storage.insert_edges(indices=new_edge_indices, attrs=new_edges_attrs)
        progress_bar.update(1)

        # STEP: Save chunks
        # Insert chunks in chunk_storage
        progress_bar.set_description("Building... [saving chunks]")
        flattened_chunks = [chunk for chunks in documents for chunk in chunks]
        await self.chunk_storage.upsert(keys=[chunk.id for chunk in flattened_chunks], values=flattened_chunks)
        progress_bar.update(1)
        progress_bar.set_description("Building [done]")

    async def upsert(
        self,
        llm: BaseLLMService,
        subgraphs: List[asyncio.Future[Optional[BaseGraphStorage[TEntity, TRelation, TId]]]],
        documents: Iterable[Iterable[TChunk]],
        show_progress: bool = True,
    ) -> None:
        nodes: Iterable[List[TEntity]]
        edges: Iterable[List[TRelation]]

        tasks = []
        failed_tasks = 0
        processed_tasks = 0

        # STEP: Extracting subgraphs
        async def _get_graphs(
            graph: Optional[BaseGraphStorage[TEntity, TRelation, TId]],
        ) -> Optional[Tuple[List[TEntity], List[TRelation]]]:
            if graph is None:
                return None

            nodes = []
            edges = []
            for i in range(await graph.node_count()):
                if (t := await graph.get_node_by_index(i)) is not None:
                    nodes.append(t)
            for i in range(await graph.edge_count()):
                if (t := await graph.get_edge_by_index(i)) is not None:
                    edges.append(t)
            return (nodes, edges)

        # Convert subgraphs to tasks with timeout
        tasks = [asyncio.create_task(_get_graphs(fgraph)) for fgraph in subgraphs]
        # Process completed tasks as they finish
        graphs = []
        for task in asyncio.as_completed(tasks, timeout=300):
            result = await task
            processed_tasks += 1
            if result is not None:
                graphs.append(result)
            else:
                failed_tasks += 1

        if len(graphs) == 0:
            logger.warning("No valid graphs were processed")
            return

        if failed_tasks > 0:
            logger.warning(f"Failed to process {failed_tasks} out of {processed_tasks} files")

        progress_bar = tqdm(total=7, disable=not show_progress, desc="Building...")
        # STEP (2): Upserting nodes and edges
        nodes, edges = zip(*graphs)
        progress_bar.set_description("Building... [upserting graphs]")

        _, upserted_nodes = await self.node_upsert_policy(
            llm,
            self.graph_storage,
            chain(*nodes),
        )
        progress_bar.update(1)
        _, _ = await self.edge_upsert_policy(llm, self.graph_storage, chain(*edges))
        progress_bar.update(1)

        # STEP (2): Computing entity embeddings
        progress_bar.set_description("Building... [computing embeddings]")
        # Insert entities in entity_storage
        embeddings = await self.embedding_service.encode(texts=[d.to_str() for _, d in upserted_nodes])
        progress_bar.update(1)
        await self.entity_storage.upsert(ids=(i for i, _ in upserted_nodes), embeddings=embeddings)
        progress_bar.update(1)

        # STEP: Entity deduplication
        # Note that get_knn will very likely return the same entity as the most similar one, so we remove it
        # when selecting the index order.
        progress_bar.set_description("Building... [entity deduplication]")
        upserted_indices = np.array([i for i, _ in upserted_nodes]).reshape(-1, 1)
        similar_indices, scores = await self.entity_storage.get_knn(embeddings, top_k=3)
        similar_indices = np.array(similar_indices)
        scores = np.array(scores)

        # Create matrix with the indices with a score higher than the threshold
        # We use the fact that similarity scores are symmetric between entity pairs,
        # so we only select half of that by index order
        similar_indices[
            (scores < self.insert_similarity_score_threshold)
            | (similar_indices <= upserted_indices)  # remove indices smaller or equal the entity
        ] = 0  # 0 can be used here (not 100% sure, but 99% sure)
        progress_bar.update(1)

        # | entity_index  | similar_indices[] |
        # |---------------|------------------------|
        # | 1             | 0, 7, 12, 0, 9         |

        # STEP: insert identity edges
        progress_bar.set_description("Building... [identity edges]")

        async def _insert_identiy_edges(
            source_index: TIndex, target_indices: npt.NDArray[np.int32]
        ) -> Iterable[Tuple[TIndex, TIndex]]:
            return [
                (source_index, idx)
                for idx in target_indices
                if idx != 0 and not await self.graph_storage.are_neighbours(source_index, idx)
            ]

        new_edge_indices = list(
            chain(
                *await asyncio.gather(*[_insert_identiy_edges(i, indices) for i, indices in enumerate(similar_indices)])
            )
        )
        new_edges_attrs: Dict[str, Any] = {
            "description": ["is"] * len(new_edge_indices),
            "chunks": [[]] * len(new_edge_indices),
        }
        await self.graph_storage.insert_edges(indices=new_edge_indices, attrs=new_edges_attrs)
        progress_bar.update(1)

        # STEP: Save chunks
        # Insert chunks in chunk_storage
        progress_bar.set_description("Building... [saving chunks]")
        flattened_chunks = [chunk for chunks in documents for chunk in chunks]
        await self.chunk_storage.upsert(keys=[chunk.id for chunk in flattened_chunks], values=flattened_chunks)
        progress_bar.update(1)
        progress_bar.set_description("Building [done]")

    async def get_context(
        self, query: str, entities: Dict[str, List[str]]
    ) -> Optional[TContext[TEntity, TRelation, THash, TChunk]]:
        if self.entity_storage.size == 0:
            return None

        try:
            query_embeddings = await self.embedding_service.encode(
                [f"{n}" for n in entities["named"]] + [f"[NONE] {n}" for n in entities["generic"]] + [query]
            )
            entity_scores: List[csr_matrix] = []
            # Similarity-search over entities
            if len(entities["named"]) > 0:
                vdb_entity_scores_by_named_entity = await self._score_entities_by_vectordb(
                    query_embeddings=query_embeddings[: len(entities["named"])],
                    top_k=1,
                    threshold=self.query_similarity_score_threshold,
                )
                entity_scores.append(vdb_entity_scores_by_named_entity)

            vdb_entity_scores_by_generic_entity_and_query = await self._score_entities_by_vectordb(
                query_embeddings=query_embeddings[len(entities["named"]) :],
                top_k=20,
                threshold=0.5,
            )
            entity_scores.append(vdb_entity_scores_by_generic_entity_and_query)

            vdb_entity_scores = vstack(entity_scores).max(axis=0)

            if isinstance(vdb_entity_scores, int) or vdb_entity_scores.nnz == 0:
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
            indices, scores = extract_sorted_scores(graph_entity_scores)
            relevant_entities: List[Tuple[TEntity, TScore]] = []
            for i, s in zip(indices, scores):
                entity = await self.graph_storage.get_node_by_index(i)
                if entity is not None:
                    relevant_entities.append((entity, s))

            # Extract relevant relationships
            relation_scores = self.relation_ranking_policy(
                await self._score_relationships_by_entities(entity_scores=graph_entity_scores)
            )

            indices, scores = extract_sorted_scores(relation_scores)
            relevant_relationships: List[Tuple[TRelation, TScore]] = []
            for i, s in zip(indices, scores):
                relationship = await self.graph_storage.get_edge_by_index(i)
                if relationship is not None:
                    relevant_relationships.append((relationship, s))

            # Extract relevant chunks
            chunk_scores = self.chunk_ranking_policy(
                await self._score_chunks_by_relations(relationships_score=relation_scores)
            )
            indices, scores = extract_sorted_scores(chunk_scores)
            relevant_chunks: List[Tuple[TChunk, TScore]] = []
            for chunk, s in zip(await self.chunk_storage.get_by_index(indices), scores):
                if chunk is not None:
                    relevant_chunks.append((chunk, s))

            return TContext(
                entities=relevant_entities,
                relations=relevant_relationships,
                chunks=relevant_chunks,
            )
        except Exception as e:
            logger.error(f"Error during scoring of chunks and relationships.\n{e}")
            raise e

    async def _get_entities_to_num_docs(self) -> Any:
        raise NotImplementedError

    async def _score_entities_by_vectordb(
        self,
        query_embeddings: Iterable[TEmbedding],
        top_k: int = 1,
        threshold: Optional[float] = None,
    ) -> csr_matrix:
        # TODO: check this
        # if top_k != 1:
        #     logger.warning(f"Top-k > 1 is not tested yet. Using top_k={top_k}.")
        if self.node_specificity:
            raise NotImplementedError("Node specificity is not supported yet.")

        all_entity_probs_by_query_entity = await self.entity_storage.score_all(
            np.array(query_embeddings), top_k=top_k, threshold=threshold
        )  # (#query_entities, #all_entities)

        # TODO: if top_k > 1, we need to aggregate the scores here
        if all_entity_probs_by_query_entity.shape[1] == 0:
            return all_entity_probs_by_query_entity
        # Normalize the scores
        all_entity_probs_by_query_entity /= all_entity_probs_by_query_entity.sum(axis=1) + 1e-8
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

        await cast(Workspace, self.workspace).with_checkpoints(_fn)

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

        await cast(Workspace, self.workspace).with_checkpoints(_fn)

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
                raw_relationships_to_chunks,
                shape=(
                    len(raw_relationships_to_chunks),
                    await self.chunk_storage.size(),
                ),
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

    async def insert_abort(self):
        """Abort the current insertion and clean up state without marking anything as complete"""
        storages: List[BaseStorage] = [
            self.graph_storage,
            self.entity_storage,
            self.chunk_storage,
            self._relationships_to_chunks,
            self._entities_to_relationships,
        ]

        # Just reset the in_progress flags without committing any changes
        for storage_inst in storages:
            storage_inst.set_in_progress(False)

    async def save_graphml(self, output_path: str) -> None:
        await self.graph_storage.save_graphml(output_path)
        logger.info(f"Graph saved to '{output_path}'.")
