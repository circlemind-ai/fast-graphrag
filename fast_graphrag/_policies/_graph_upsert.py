import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import chain
from typing import Counter, Dict, Iterable, List, Optional, Set, Tuple, Union

from fast_graphrag._llm._base import format_and_send_prompt
from fast_graphrag._llm._llm_openai import BaseLLMService
from fast_graphrag._models import TEditRelationList, TEntityDescription
from fast_graphrag._prompt import PROMPTS
from fast_graphrag._storage._base import BaseGraphStorage
from fast_graphrag._types import GTEdge, GTId, GTNode, TEntity, THash, TId, TIndex, TRelation
from fast_graphrag._utils import logger

from ._base import BaseEdgeUpsertPolicy, BaseGraphUpsertPolicy, BaseNodeUpsertPolicy


async def summarize_entity_description(
    prompt: str, description: str, llm: BaseLLMService, max_tokens: Optional[int] = None
) -> str:
    """Summarize the given entity description."""
    if max_tokens is not None:
        raise NotImplementedError("Summarization with max tokens is not yet supported.")
    # Extract entities and relationships
    system_key = prompt + '_system'
    # Split system prompt/initial prompt pair
    if system_key in PROMPTS:
        # Use separate system and prompt entries if available.
        entity_description_summarization_system = PROMPTS[system_key]
        entity_description_summarization_prompt = PROMPTS[prompt + '_prompt']

        formatted_system = entity_description_summarization_system.format(description=description)
        formatted_prompt = entity_description_summarization_prompt.format(description=description)

        new_description, _ = await llm.send_message(
            system_prompt=formatted_system,
            prompt=formatted_prompt,
            response_model=TEntityDescription,
        )
    else:
        # Single prompt summarization
        entity_description_summarization_prompt = PROMPTS[prompt]
        formatted_entity_description_summarization_prompt = entity_description_summarization_prompt.format(
            description=description
        )
        new_description, _ = await llm.send_message(
            prompt=formatted_entity_description_summarization_prompt,
            response_model=TEntityDescription,
        )

    return new_description.description


####################################################################################################
# DEFAULT GRAPH UPSERT POLICIES
####################################################################################################


@dataclass
class DefaultNodeUpsertPolicy(BaseNodeUpsertPolicy[GTNode, GTId]):
    async def __call__(
        self, llm: BaseLLMService, target: BaseGraphStorage[GTNode, GTEdge, GTId], source_nodes: Iterable[GTNode]
    ) -> Tuple[BaseGraphStorage[GTNode, GTEdge, GTId], Iterable[Tuple[TIndex, GTNode]]]:
        upserted: Dict[TIndex, GTNode] = {}
        for node in source_nodes:
            _, index = await target.get_node(node)
            if index is not None:
                await target.upsert_node(node=node, node_index=index)
            else:
                index = await target.upsert_node(node=node, node_index=None)
            upserted[index] = node

        return target, upserted.items()


@dataclass
class DefaultEdgeUpsertPolicy(BaseEdgeUpsertPolicy[GTEdge, GTId]):
    async def __call__(
        self, llm: BaseLLMService, target: BaseGraphStorage[GTNode, GTEdge, GTId], source_edges: Iterable[GTEdge]
    ) -> Tuple[BaseGraphStorage[GTNode, GTEdge, GTId], Iterable[Tuple[TIndex, GTEdge]]]:
        indices = await target.insert_edges(source_edges)

        r: Iterable[Tuple[TIndex, GTEdge]]
        if len(indices):
            r = zip(indices, source_edges)
        else:
            r = []
        return target, r


@dataclass
class DefaultGraphUpsertPolicy(BaseGraphUpsertPolicy[GTNode, GTEdge, GTId]):  # noqa: N801
    async def __call__(
        self,
        llm: BaseLLMService,
        target: BaseGraphStorage[GTNode, GTEdge, GTId],
        source_nodes: Iterable[GTNode],
        source_edges: Iterable[GTEdge],
    ) -> Tuple[
        BaseGraphStorage[GTNode, GTEdge, GTId],
        Iterable[Tuple[TIndex, GTNode]],
        Iterable[Tuple[TIndex, GTEdge]],
    ]:
        target, upserted_nodes = await self._nodes_upsert(llm, target, source_nodes)
        target, upserted_edges = await self._edges_upsert(llm, target, source_edges)

        return target, upserted_nodes, upserted_edges


####################################################################################################
# NODE UPSERT POLICIES
####################################################################################################


@dataclass
class NodeUpsertPolicy_SummarizeDescription(BaseNodeUpsertPolicy[TEntity, TId]):  # noqa: N801
    @dataclass
    class Config:
        max_node_description_size: int = field(default=512)
        node_summarization_ratio: float = field(default=0.5)
        node_summarization_prompt: str = field(default="summarize_entity_descriptions")
        is_async: bool = field(default=True)

    config: Config = field(default_factory=Config)

    async def __call__(
        self, llm: BaseLLMService, target: BaseGraphStorage[TEntity, GTEdge, TId], source_nodes: Iterable[TEntity]
    ) -> Tuple[BaseGraphStorage[TEntity, GTEdge, TId], Iterable[Tuple[TIndex, TEntity]]]:
        upserted: List[Tuple[TIndex, TEntity]] = []

        async def _upsert_node(node_id: TId, nodes: List[TEntity]) -> Optional[Tuple[TIndex, TEntity]]:
            existing_node, index = await target.get_node(node_id)
            if existing_node:
                nodes.append(existing_node)

            # Resolve descriptions
            node_description = "\n".join((node.description for node in nodes))

            if len(node_description) > self.config.max_node_description_size:
                node_description = await summarize_entity_description(
                    self.config.node_summarization_prompt,
                    node_description,
                    llm,
                    # int(
                    #     self.config.max_node_description_size
                    #     * self.config.node_summarization_ratio
                    #     / TOKEN_TO_CHAR_RATIO
                    # ),
                )

            # Resolve types (pick most frequent)
            node_type = Counter((node.type for node in nodes)).most_common(1)[0][0]

            node = TEntity(name=node_id, description=node_description, type=node_type)
            index = await target.upsert_node(node=node, node_index=index)

            upserted.append((index, node))

        # Group nodes by name
        grouped_nodes: Dict[TId, List[TEntity]] = defaultdict(lambda: [])
        for node in source_nodes:
            grouped_nodes[node.name].append(node)

        if self.config.is_async:
            node_upsert_tasks = (_upsert_node(node_id, nodes) for node_id, nodes in grouped_nodes.items())
            await asyncio.gather(*node_upsert_tasks)
        else:
            for node_id, nodes in grouped_nodes.items():
                await _upsert_node(node_id, nodes)

        return target, upserted


####################################################################################################
# EDGE UPSERT POLICIES
####################################################################################################


@dataclass
class EdgeUpsertPolicy_UpsertIfValidNodes(BaseEdgeUpsertPolicy[TRelation, TId]):  # noqa: N801
    @dataclass
    class Config:
        is_async: bool = field(default=True)

    config: Config = field(default_factory=Config)

    async def __call__(
        self, llm: BaseLLMService, target: BaseGraphStorage[GTNode, TRelation, TId], source_edges: Iterable[TRelation]
    ) -> Tuple[BaseGraphStorage[GTNode, TRelation, TId], Iterable[Tuple[TIndex, TRelation]]]:
        new_edges: List[TRelation] = []

        async def _upsert_edge(edge: TRelation) -> Optional[Tuple[TIndex, TRelation]]:
            source_node, _ = await target.get_node(edge.source)
            target_node, _ = await target.get_node(edge.target)

            if source_node and target_node:
                new_edges.append(edge)

        if self.config.is_async:
            edge_upsert_tasks = (_upsert_edge(edge) for edge in source_edges)
            await asyncio.gather(*edge_upsert_tasks)
        else:
            for edge in source_edges:
                await _upsert_edge(edge)
        indices = await target.insert_edges(new_edges)

        r: Iterable[Tuple[TIndex, TRelation]]
        if len(indices):
            r = zip(indices, new_edges)
        else:
            r = []

        return target, r


@dataclass
class EdgeUpsertPolicy_UpsertValidAndMergeSimilarByLLM(BaseEdgeUpsertPolicy[TRelation, TId]):  # noqa: N801
    @dataclass
    class Config:
        edge_merge_threshold: int = field(default=5)
        is_async: bool = field(default=True)

    config: Config = field(default_factory=Config)

    async def _upsert_edge(
        self,
        llm: BaseLLMService,
        target: BaseGraphStorage[GTNode, TRelation, TId],
        edges: List[TRelation],
        source_entity: TId,
        target_entity: TId,
    ) -> Tuple[List[Tuple[TIndex, TRelation]], List[TRelation], List[TIndex]]:
        existing_edges = list(await target.get_edges(source_entity, target_entity))

        # Check if we need to run edges maintenance
        if (len(existing_edges) + len(edges)) > self.config.edge_merge_threshold:
            upserted_eges, new_edges, to_delete_edges = await self._merge_similar_edges(
                llm, target, existing_edges, edges
            )
        else:
            upserted_eges = []
            new_edges = edges
            to_delete_edges = []

        return upserted_eges, new_edges, to_delete_edges

    async def _merge_similar_edges(
        self,
        llm: BaseLLMService,
        target: BaseGraphStorage[GTNode, TRelation, TId],
        existing_edges: List[Tuple[TRelation, TIndex]],
        edges: List[TRelation],
    ) -> Tuple[List[Tuple[TIndex, TRelation]], List[TRelation], List[TIndex]]:
        """Merge similar edges between the same pair of nodes.

        Args:
            llm (BaseLLMService): The language model that is called to determine the similarity between edges.
            target (BaseGraphStorage[GTNode, TRelation, TId]): the graph storage to upsert the edges to.
            existing_edges (List[Tuple[TRelation, TIndex]]): list of existing edges in the main graph storage.
            edges (List[TRelation]): list of new edges to be upserted.

        Returns:
            Tuple[List[Tuple[TIndex, TRelation]], List[TIndex]]: return the pairs of inserted (index, edge),
            the new edges that were not merged, and the indices of the edges that are to be deleted.
        """
        updated_edges: List[Tuple[TIndex, TRelation]] = []
        new_edges: List[TRelation] = []
        map_incremental_to_edge: Dict[int, Tuple[TRelation, Union[TIndex, None]]] = {
            **dict(enumerate(existing_edges)),
            **{idx + len(existing_edges): (edge, None) for idx, edge in enumerate(edges)},
        }

        # Extract entities and relationships
        edge_grouping, _ = await format_and_send_prompt(
            prompt_key="edges_group_similar",
            llm=llm,
            format_kwargs={
                "edge_list": "\n".join(
                    (f"{idx}, {edge.description}" for idx, (edge, _) in map_incremental_to_edge.items())
                )
            },
            response_model=TEditRelationList,
        )

        visited_edges: Dict[TIndex, Union[TIndex, None]] = {}
        for edges_group in edge_grouping.groups:
            relation_indices = [
                index
                for index in edges_group.ids
                if index < len(existing_edges) + len(edges)  # Only consider valid indices
            ]
            if len(relation_indices) < 2:
                logger.info("LLM returned invalid index for edge maintenance, ignoring.")
                continue

            chunks: Set[THash] = set()

            for second in relation_indices[1:]:
                edge, index = map_incremental_to_edge[second]

                # Set visited edges only the first time we see them.
                # In this way, if an existing edge is marked for "not deletion" later, we do not overwrite it.
                if second not in visited_edges:
                    visited_edges[second] = index
                if edge.chunks:
                    chunks.update(edge.chunks)

            first_index = relation_indices[0]
            edge, index = map_incremental_to_edge[first_index]
            edge.description = edges_group.description
            visited_edges[first_index] = None  # None means it was visited but not marked for deletion.
            if edge.chunks:
                chunks.update(edge.chunks)
            edge.chunks = list(chunks)
            if index is not None:
                updated_edges.append((await target.upsert_edge(edge, index), edge))
            else:
                new_edges.append(edge)

        for idx, edge in enumerate(edges):
            # If the edge was not visited, it means it was not grouped and must be inserted as new.
            if idx + len(existing_edges) not in visited_edges:
                new_edges.append(edge)
                # upserted_eges.append((await target.upsert_edge(edge, None), edge))

        # Only existing edges that were marked for deletion have non-None value which corresponds to their real index.
        return updated_edges, new_edges, [v for v in visited_edges.values() if v is not None]

    async def __call__(
        self, llm: BaseLLMService, target: BaseGraphStorage[GTNode, TRelation, TId], source_edges: Iterable[TRelation]
    ) -> Tuple[BaseGraphStorage[GTNode, TRelation, TId], Iterable[Tuple[TIndex, TRelation]]]:
        grouped_edges: Dict[Tuple[TId, TId], List[TRelation]] = defaultdict(lambda: [])
        upserted_edges: Tuple[List[Tuple[TIndex, TRelation]], ...] = ()
        new_edges: Tuple[List[TRelation], ...] = ()
        to_delete_edges: Tuple[List[TIndex], ...] = ()
        for edge in source_edges:
            grouped_edges[(edge.source, edge.target)].append(edge)

        if self.config.is_async:
            edge_upsert_tasks = (
                self._upsert_edge(llm, target, edges, source_entity, target_entity)
                for (source_entity, target_entity), edges in grouped_edges.items()
            )
            tasks = await asyncio.gather(*edge_upsert_tasks)
            if len(tasks):
                upserted_edges, new_edges, to_delete_edges = zip(*tasks)
        else:
            tasks = [
                await self._upsert_edge(llm, target, edges, source_entity, target_entity)
                for (source_entity, target_entity), edges in grouped_edges.items()
            ]
            if len(tasks):
                upserted_edges, new_edges, to_delete_edges = zip(*tasks)
        await target.delete_edges_by_index(tuple(chain(*to_delete_edges)))
        new_indices = await target.insert_edges(tuple(chain(*new_edges)))
        return target, chain(*upserted_edges, zip(new_indices, chain(*new_edges)))
