import asyncio
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Counter, Dict, Iterable, List, Optional, Tuple
from pathlib import Path

from fast_graphrag._services._batch_information_extraction import (
    extract_json_from_llm_response,
)

from fast_graphrag._models import (
    TBedrockBatchInput,
    TBedrockBatchRequest,
    TClaudeContentBlock,
    TClaudeMessage,
)
from fast_graphrag._llm._llm_openai import BaseLLMService
from fast_graphrag._prompt import PROMPTS
from fast_graphrag._storage._base import BaseGraphStorage
from fast_graphrag._types import (
    GTEdge,
    TEntity,
    TId,
    TIndex,
)

from itertools import chain
from typing import Set, Union

from ._base import (
    BatchBaseNodeUpsertPolicy,
    BatchBaseNodeUpsertPromptPolicy,
    BatchBaseEdgeUpsertPolicy,
    BatchBaseEdgeUpsertPromptPolicy,
)
from fast_graphrag._models import TEditRelationList
from fast_graphrag._types import (
    GTNode,
    THash,
    TRelation,
)
from fast_graphrag._utils import logger


TOOLS = [
    {
        "name": "summarize_entity_description",
        "description": "Output a summarized description, removing redundant and generic information.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summarizedDescription": {
                    "type": "string",
                    "description": "The summarized description",
                }
            },
            "required": ["summarizedDescription"],
        },
    },
    {
        "name": "group_edges_similar",
        "description": "Identify any facts that should be grouped together as they contain similar or duplicated information",
        "input_schema": {
            "type": "object",
            "$defs": {
                "grouping": {
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "A short description of the fact",
                        },
                        "ids": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "A list of IDs for facts that match this description. Each ID must have been in the input prompt",
                        },
                    },
                    "required": ["description", "ids"],
                }
            },
            "properties": {
                "grouped_facts": {
                    "type": "array",
                    "description": "The list of grouped facts",
                    "items": {"$ref": "#/$defs/grouping"},
                }
            },
            "required": ["grouped_facts"],
        },
    },
]


def generate_stable_node_prompt_id(node_id: TId) -> str:
    desc_hash = hashlib.sha256(node_id.encode()).hexdigest()[:11]
    return f"{desc_hash}"


def generate_stable_edge_prompt_id(source_entity: TId, target_entity: TId) -> str:
    pair_string = f"{source_entity}:{target_entity}"
    pair_hash = hashlib.sha256(pair_string.encode()).hexdigest()[:11]
    return pair_hash


@dataclass
class NodeUpsertPolicy_SummarizeDescription_PreparePrompts(BatchBaseNodeUpsertPromptPolicy[TEntity, TId]):
    @dataclass
    class Config:
        max_node_description_size: int = field(default=512)
        node_summarization_ratio: float = field(default=0.5)
        node_summarization_prompt: str = field(default=PROMPTS["summarize_entity_descriptions"])
        is_async: bool = field(default=True)

    config: Config = field(default_factory=Config)

    def _write_prompt(
        self,
        prompt: str,
        description: str,
        node_id: TId,
        summarize_node_prompt_path: Path,
    ):
        prompt_id = generate_stable_node_prompt_id(node_id)
        with open(summarize_node_prompt_path, "a") as f:
            f.write(
                TBedrockBatchRequest(
                    recordId=prompt_id,
                    modelInput=TBedrockBatchInput(
                        anthropic_version="bedrock-2023-05-31",
                        max_tokens=4096,
                        tool_choice={
                            "type": "tool",
                            "name": "summarize_entity_description",
                        },
                        tools=TOOLS,
                        messages=[
                            TClaudeMessage(
                                role="user",
                                content=[
                                    TClaudeContentBlock(
                                        type="text",
                                        text=prompt.format(description=description),
                                    )
                                ],
                            )
                        ],
                    ),
                ).model_dump_json()
                + "\n"
            )

    async def __call__(
        self,
        target: BaseGraphStorage[TEntity, GTEdge, TId],
        source_nodes: Iterable[TEntity],
        summarize_node_prompt_path: Optional[Path] = None,
    ):
        async def _write_node_prompt(node_id: TId, nodes: List[TEntity]) -> Optional[Tuple[TIndex, TEntity]]:
            existing_node, index = await target.get_node(node_id)
            if existing_node:
                nodes.append(existing_node)

            node_description = "\n".join((node.description for node in nodes))

            if len(node_description) > self.config.max_node_description_size:
                self._write_prompt(
                    self.config.node_summarization_prompt,
                    node_description,
                    node_id,
                    summarize_node_prompt_path,
                )

        open(summarize_node_prompt_path, "w").close()

        # Group nodes by name
        grouped_nodes: Dict[TId, List[TEntity]] = defaultdict(list)
        for node in source_nodes:
            grouped_nodes[node.name].append(node)

        if self.config.is_async:
            node_upsert_tasks = (_write_node_prompt(node_id, nodes) for node_id, nodes in grouped_nodes.items())
            await asyncio.gather(*node_upsert_tasks)
        else:
            for node_id, nodes in grouped_nodes.items():
                await _write_node_prompt(node_id, nodes)


@dataclass
class NodeUpsertPolicy_SummarizeDescription_BatchAsync(BatchBaseNodeUpsertPolicy[TEntity, TId]):
    @dataclass
    class Config:
        max_node_description_size: int = field(default=512)
        node_summarization_ratio: float = field(default=0.5)
        is_async: bool = field(default=True)

    config: Config = field(default_factory=Config)

    async def __call__(
        self,
        llm: BaseLLMService,
        target: BaseGraphStorage[TEntity, GTEdge, TId],
        source_nodes: Iterable[TEntity],
        summarize_node_output: dict,
    ) -> Tuple[BaseGraphStorage[TEntity, GTEdge, TId], Iterable[Tuple[TIndex, TEntity]]]:
        upserted: List[Tuple[TIndex, TEntity]] = []

        async def _upsert_node(node_id: TId, nodes: List[TEntity]) -> Optional[Tuple[TIndex, TEntity]]:
            existing_node, index = await target.get_node(node_id)
            if existing_node:
                nodes.append(existing_node)

            # Resolve descriptions
            node_description = "\n".join((node.description for node in nodes))

            if len(node_description) > self.config.max_node_description_size:
                prompt_id = generate_stable_node_prompt_id(node_id)
                try:
                    summarized_output = summarize_node_output[prompt_id]
                    # TODO: convert to dataclass?
                    node_description = extract_json_from_llm_response(summarized_output).get("summarizedDescription")
                except KeyError:
                    print(f"Failed to find prompt id {prompt_id} in summarized node output")
                except Exception:
                    print(f"Failed to extract json {prompt_id} in summarized node output")

            # Resolve types (pick most frequent)
            node_type = Counter((node.type for node in nodes)).most_common(1)[0][0]

            node = TEntity(name=node_id, description=node_description, type=node_type)
            index = await target.upsert_node(node=node, node_index=index)

            return (index, node)

        # Group nodes by name
        grouped_nodes: Dict[TId, List[TEntity]] = defaultdict(list)
        for node in source_nodes:
            grouped_nodes[node.name].append(node)

        total = len(grouped_nodes)

        if self.config.is_async:
            pbar = tqdm(total=len(grouped_nodes), desc="Upserting nodes")

            async def _upsert_with_progress(node_id: TId, nodes: List[TEntity]):
                result = await _upsert_node(node_id, nodes)
                pbar.update(1)
                return result

            results = await asyncio.gather(
                *[_upsert_with_progress(node_id, nodes) for node_id, nodes in grouped_nodes.items()]
            )

            upserted.extend(results)
            pbar.close()
        else:
            # Process sequentially with progress bar
            for node_id, nodes in tqdm(grouped_nodes.items(), total=total, desc="Upserting nodes"):
                result = await _upsert_node(node_id, nodes)
                upserted.append(result)

        return target, upserted


@dataclass
class EdgeUpsertPolicy_UpsertValidAndMergeSimilarByLLM_PreparePrompts(BatchBaseEdgeUpsertPromptPolicy[TRelation, TId]):  # noqa: N801
    @dataclass
    class Config:
        edge_merge_threshold: int = field(default=5)
        is_async: bool = field(default=True)

    config: Config = field(default_factory=Config)

    async def _write_prompt(
        self,
        existing_edges: List[Tuple[TRelation, TIndex]],
        edges: List[TRelation],
        prompt_id: str,
        summarize_edge_prompt_path: Optional[Path] = None,
    ) -> Tuple[List[Tuple[TIndex, TRelation]], List[TRelation], List[TIndex]]:
        map_incremental_to_edge: Dict[int, Tuple[TRelation, Union[TIndex, None]]] = {
            **dict(enumerate(existing_edges)),
            **{idx + len(existing_edges): (edge, None) for idx, edge in enumerate(edges)},
        }

        edge_list = "\n".join((f"{idx}, {edge.description}" for idx, (edge, _) in map_incremental_to_edge.items()))
        with open(summarize_edge_prompt_path, "a") as f:
            f.write(
                TBedrockBatchRequest(
                    recordId=prompt_id,
                    modelInput=TBedrockBatchInput(
                        anthropic_version="bedrock-2023-05-31",
                        max_tokens=4096,
                        tool_choice={
                            "type": "tool",
                            "name": "group_edges_similar",
                        },
                        tools=TOOLS,
                        messages=[
                            TClaudeMessage(
                                role="user",
                                content=[
                                    TClaudeContentBlock(
                                        type="text",
                                        text=PROMPTS["edges_group_similar"].format(edge_list=edge_list),
                                    )
                                ],
                            )
                        ],
                    ),
                ).model_dump_json()
                + "\n"
            )
        return

    async def _prepare_edge_prompt(
        self,
        target: BaseGraphStorage[GTNode, TRelation, TId],
        edges: List[TRelation],
        source_entity: TId,
        target_entity: TId,
        summarize_edge_prompt_path: Optional[Path] = None,
    ):
        existing_edges = list(await target.get_edges(source_entity, target_entity))

        if (len(existing_edges) + len(edges)) > self.config.edge_merge_threshold:
            await self._write_prompt(
                existing_edges,
                edges,
                generate_stable_edge_prompt_id(source_entity, target_entity),
                summarize_edge_prompt_path,
            )
        return

    async def __call__(
        self,
        target: BaseGraphStorage[GTNode, TRelation, TId],
        source_edges: Iterable[TRelation],
        summarize_edge_prompt_path: Optional[Path] = None,
    ):
        grouped_edges: Dict[Tuple[TId, TId], List[TRelation]] = defaultdict(lambda: [])
        for edge in source_edges:
            grouped_edges[(edge.source, edge.target)].append(edge)

        # clear the file first to remove previous output
        open(summarize_edge_prompt_path, "w").close()

        if self.config.is_async:
            edge_upsert_tasks = (
                self._prepare_edge_prompt(
                    target,
                    edges,
                    source_entity,
                    target_entity,
                    summarize_edge_prompt_path,
                )
                for (source_entity, target_entity), edges in grouped_edges.items()
            )
            await asyncio.gather(*edge_upsert_tasks)
        else:
            [
                await self._prepare_edge_prompt(
                    target,
                    edges,
                    source_entity,
                    target_entity,
                    summarize_edge_prompt_path,
                )
                for (source_entity, target_entity), edges in grouped_edges.items()
            ]


@dataclass
class EdgeUpsertPolicy_UpsertValidAndMergeSimilarByLLM_BatchAsync(BatchBaseEdgeUpsertPolicy[TRelation, TId]):  # noqa: N801
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
        summarize_edge_output: dict,
    ) -> Tuple[List[Tuple[TIndex, TRelation]], List[TRelation], List[TIndex]]:
        existing_edges = list(await target.get_edges(source_entity, target_entity))

        # Check if we need to run edges maintenance
        if (len(existing_edges) + len(edges)) > self.config.edge_merge_threshold:
            upserted_eges, new_edges, to_delete_edges = await self._merge_similar_edges(
                llm,
                target,
                existing_edges,
                edges,
                generate_stable_edge_prompt_id(source_entity, target_entity),
                summarize_edge_output,
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
        prompt_id: str,
        summarize_edge_output: dict,
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

        try:
            json = extract_json_from_llm_response(summarize_edge_output[prompt_id])
            edge_grouping = TEditRelationList.parse_obj(json)
        except KeyError:
            print(f"Failed to find prompt id {prompt_id} in summarized edge output")
            edge_grouping = TEditRelationList(grouped_facts=[])
        except Exception:
            # TODO send to LLM and ask to output valid JSON
            print(f"Cannot parse {json} in summarized edge output")
            edge_grouping = TEditRelationList(grouped_facts=[])

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
        return (
            updated_edges,
            new_edges,
            [v for v in visited_edges.values() if v is not None],
        )

    async def __call__(
        self,
        llm: BaseLLMService,
        target: BaseGraphStorage[GTNode, TRelation, TId],
        source_edges: Iterable[TRelation],
        summarize_edge_output: dict,
    ) -> Tuple[BaseGraphStorage[GTNode, TRelation, TId], Iterable[Tuple[TIndex, TRelation]]]:
        grouped_edges: Dict[Tuple[TId, TId], List[TRelation]] = defaultdict(lambda: [])
        upserted_edges: List[List[Tuple[TIndex, TRelation]]] = []
        new_edges: List[List[TRelation]] = []
        to_delete_edges: List[List[TIndex]] = []

        for edge in source_edges:
            grouped_edges[(edge.source, edge.target)].append(edge)

        if self.config.is_async:
            pbar = tqdm(total=len(grouped_edges), desc="Upserting edges")

            async def _upsert_edge_with_progress(source_entity: TId, target_entity: TId, edges: List[TRelation]):
                result = await self._upsert_edge(
                    llm,
                    target,
                    edges,
                    source_entity,
                    target_entity,
                    summarize_edge_output,
                )
                pbar.update(1)
                return result

            tasks = await asyncio.gather(
                *[
                    _upsert_edge_with_progress(source_entity, target_entity, edges)
                    for (source_entity, target_entity), edges in grouped_edges.items()
                ]
            )

            pbar.close()

            if len(tasks):
                upserted_edges, new_edges, to_delete_edges = zip(*tasks)
        else:
            tasks = []
            for (source_entity, target_entity), edges in tqdm(grouped_edges.items(), desc="Upserting edges"):
                result = await self._upsert_edge(
                    llm,
                    target,
                    edges,
                    source_entity,
                    target_entity,
                    summarize_edge_output,
                )
                tasks.append(result)

            if len(tasks):
                upserted_edges, new_edges, to_delete_edges = zip(*tasks)

        await target.delete_edges_by_index(chain(*to_delete_edges))
        new_indices = await target.insert_edges(chain(*new_edges))
        return target, chain(*upserted_edges, zip(new_indices, chain(*new_edges)))
