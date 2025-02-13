"""Entity-Relationship extraction module."""

import asyncio
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional

from pydantic import BaseModel, Field

from fast_graphrag._llm import BaseLLMService, format_and_send_prompt
from fast_graphrag._models import TQueryEntities
from fast_graphrag._storage._base import BaseGraphStorage
from fast_graphrag._storage._gdb_igraph import IGraphStorage, IGraphStorageConfig
from fast_graphrag._types import GTId, TChunk, TEntity, TGraph, TRelation
from fast_graphrag._utils import logger

from ._base import BaseInformationExtractionService


class TGleaningStatus(BaseModel):
    status: Literal["done", "continue"] = Field(
        description="done if all entities and relationship have been extracted, continue otherwise"
    )


@dataclass
class DefaultInformationExtractionService(BaseInformationExtractionService[TChunk, TEntity, TRelation, GTId]):
    """Default entity and relationship extractor."""

    def extract(
        self,
        llm: BaseLLMService,
        documents: Iterable[Iterable[TChunk]],
        prompt_kwargs: Dict[str, str],
        entity_types: List[str],
    ) -> List[asyncio.Future[Optional[BaseGraphStorage[TEntity, TRelation, GTId]]]]:
        """Extract both entities and relationships from the given data."""
        return [
            asyncio.create_task(
                self._extract(llm, document, prompt_kwargs, entity_types),
            )
            for document in documents
        ]

    async def extract_entities_from_query(
        self, llm: BaseLLMService, query: str, prompt_kwargs: Dict[str, str]
    ) -> Dict[str, List[str]]:
        """Extract entities from the given query."""
        prompt_kwargs["query"] = query
        entities, _ = await format_and_send_prompt(
            prompt_key="entity_extraction_query",
            llm=llm,
            format_kwargs=prompt_kwargs,
            response_model=TQueryEntities,
        )

        return {"named": entities.named, "generic": entities.generic}

    async def _extract(
        self,
        llm: BaseLLMService,
        chunks: Iterable[TChunk],
        prompt_kwargs: Dict[str, str],
        entity_types: List[str],
    ) -> Optional[BaseGraphStorage[TEntity, TRelation, GTId]]:
        """Extract both entities and relationships from the given chunks."""
        print(f"_extract chunk count {len(chunks)}")
        chunk_graphs = await asyncio.gather(
            *[self._extract_from_chunk(llm, chunk, prompt_kwargs, entity_types) for chunk in chunks],
            return_exceptions=True,
        )

        for result in chunk_graphs:
            if isinstance(result, Exception):
                logger.error(f"Error during extract_from_chunk: {result}")
                raise result

        return await self._merge(llm, chunk_graphs)

    async def _gleaning(
        self, llm: BaseLLMService, initial_graph: TGraph, history: list[dict[str, str]]
    ) -> Optional[TGraph]:
        """Do gleaning steps until the llm says we are done or we reach the max gleaning steps."""
        # Prompts
        current_graph = initial_graph

        for gleaning_count in range(self.max_gleaning_steps):
            # Do gleaning step
            gleaning_result, history = await format_and_send_prompt(
                prompt_key="entity_relationship_continue_extraction",
                llm=llm,
                format_kwargs={},
                response_model=TGraph,
                history_messages=history,
            )

            # Combine new entities, relationships with previously obtained ones
            current_graph.entities.extend(gleaning_result.entities)
            current_graph.relationships.extend(gleaning_result.relationships)

            # Stop gleaning if we don't need to keep going
            if gleaning_count == self.max_gleaning_steps - 1:
                break

            # Ask llm if we are done extracting entities and relationships
            gleaning_status, _ = await format_and_send_prompt(
                prompt_key="entity_relationship_gleaning_done_extraction",
                llm=llm,
                format_kwargs={},
                response_model=TGleaningStatus,
                history_messages=history,
            )

            # If we are done parsing, stop gleaning
            if gleaning_status.status == Literal["done"]:
                break

        return current_graph

    async def _extract_from_chunk(
        self,
        llm: BaseLLMService,
        chunk: TChunk,
        prompt_kwargs: Dict[str, str],
        entity_types: List[str],
    ) -> TGraph:
        """Extract entities and relationships from the given chunk."""
        prompt_kwargs["input_text"] = chunk.content

        print(f"Starting extraction from chunk {chunk.id}")

        chunk_graph, history = await format_and_send_prompt(
            prompt_key="entity_relationship_extraction",
            llm=llm,
            format_kwargs=prompt_kwargs,
            response_model=TGraph,
        )

        # Do gleaning
        chunk_graph_with_gleaning = await self._gleaning(llm, chunk_graph, history)
        if chunk_graph_with_gleaning:
            chunk_graph = chunk_graph_with_gleaning

        _clean_entity_types = [re.sub("[ _]", "", entity_type).upper() for entity_type in entity_types]
        for entity in chunk_graph.entities:
            if re.sub("[ _]", "", entity.type).upper() not in _clean_entity_types:
                entity.type = "UNKNOWN"

        # Assign chunk ids to relationships
        for relationship in chunk_graph.relationships:
            relationship.chunks = [chunk.id]

        return chunk_graph

    async def _merge(self, llm: BaseLLMService, graphs: List[TGraph]) -> BaseGraphStorage[TEntity, TRelation, GTId]:
        """Merge the given graphs into a single graph storage."""
        graph_storage = IGraphStorage[TEntity, TRelation, GTId](config=IGraphStorageConfig(TEntity, TRelation))

        await graph_storage.insert_start()

        # This is synchronous since each sub graph is inserted into the graph storage and conflicts are resolved
        for graph in graphs:
            await self.graph_upsert(llm, graph_storage, graph.entities, graph.relationships)
        await graph_storage.insert_done()

        return graph_storage
