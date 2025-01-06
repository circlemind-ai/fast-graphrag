"""Entity-Relationship extraction module using async batch Bedrock API."""

import asyncio
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional
from pathlib import Path

import numpy as np
import json
import hashlib

from pydantic import BaseModel, Field

from fast_graphrag._llm import BaseLLMService, format_and_write_prompt
from fast_graphrag._models import (
    TClaudeContentBlock,
    TClaudeMessage,
    TBedrockBatchInput,
    TBedrockBatchRequest,
    TBedrockBatchResponse,
    TBedrockBatchOutput,
)
from fast_graphrag._storage._base import BaseGraphStorage
from fast_graphrag._storage._gdb_igraph import IGraphStorage, IGraphStorageConfig
from fast_graphrag._types import (
    GTId,
    TChunk,
    TEntity,
    TGraph,
    TRelation,
)

from ._base import BaseInformationExtractionService

extract_count = 0
extract_error_count = 0
glean_count = 0
glean_error_count = 0


class TGleaningStatus(BaseModel):
    status: Literal["done", "continue"] = Field(
        description="done if all entities and relationship have been extracted, continue otherwise"
    )


def generate_stable_id(chunk_index: int = 0) -> str:
    """Generate an 11-character stable alphanumeric ID with minimal collision risk.

    Uses SHA-256 hash encoded in base36, truncated or padded to exactly 11 characters.
    """
    input_string = f"{chunk_index}"
    hash_object = hashlib.sha256(input_string.encode())
    # Convert the entire hash to an integer
    n = int.from_bytes(hash_object.digest(), byteorder="big")
    # Convert to base36 (alphanumeric, case-insensitive)
    base36 = np.base_repr(n, 36)
    # Take first 11 characters if longer, or pad with zeros if shorter
    if len(base36) > 11:
        return base36[:11]
    return base36.zfill(11)


def extract_json_from_llm_response(response: TBedrockBatchResponse):
    for block in response.modelOutput.content:
        if block.type == "tool_result":
            return json.loads(block.content)
        if block.type == "tool_use":
            return block.input
    for block in response.modelOutput.content:
        if block.type == "text":
            text_response = block.text
            start = text_response.find("{")
            end = text_response.rfind("}") + 1
            json_text = text_response[start:end]
            try:
                return json.loads(json_text)
            except Exception:
                return None
    return None


TOOLS = [
    {
        "name": "extract_entity_relation",
        "description": "Extract entity and relation definitions from the text",
        "input_schema": {
            "type": "object",
            "$defs": {
                "relationship": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "An entity that is the source/originator of the relationship. Must be present in the entities array.",
                        },
                        "target": {
                            "type": "string",
                            "description": "An entity that is the object of the relationship. Must be present in the entities array.",
                        },
                        "desc": {
                            "type": "string",
                            "description": "A brief (fewer than 10 words) explanation of how source relates to target",
                        },
                    },
                    "required": ["source", "target", "desc"],
                }
            },
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The name of the entity, uniquely identifying it",
                            },
                            "type": {
                                "type": "string",
                                "description": "The type of the entity. Should be one of the Types provided in prompt in INPUT_DATA",
                            },
                            "desc": {
                                "type": "string",
                                "description": "A short (< 10 words), self-contained description of the entity",
                            },
                        },
                        "required": ["name", "type", "desc"],
                    },
                },
                "relationships": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/relationship"},
                },
                "other_relationships": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/relationship"},
                },
            },
            "required": ["entities", "relationships", "other_relationships"],
        },
    }
]


@dataclass
class BatchInformationExtractionService(BaseInformationExtractionService[TChunk, TEntity, TRelation, GTId]):
    """Batch entity and relationship extractor."""

    def prepare_batch_extraction_prompt(
        self,
        documents: Iterable[Iterable[TChunk]],
        output_path: Path,
        prompt_kwargs: Dict[str, str],
        entity_types: List[str],
    ) -> bool:
        """Extract both entities and relationships from the given data."""
        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:
            try:
                for document in documents:
                    for chunk in document:
                        prompt_kwargs["input_text"] = chunk.content

                        record_id = generate_stable_id(chunk.id)

                        prompt = format_and_write_prompt(
                            prompt_key="entity_relationship_extraction",
                            format_kwargs=prompt_kwargs,
                        )

                        # TODO: remove Bedrock/Claude specific API here
                        f.write(
                            TBedrockBatchRequest(
                                recordId=record_id,
                                modelInput=TBedrockBatchInput(
                                    anthropic_version="bedrock-2023-05-31",
                                    max_tokens=4096,
                                    tools=TOOLS,
                                    tool_choice={
                                        "type": "tool",
                                        "name": "extract_entity_relation",
                                    },
                                    messages=[
                                        TClaudeMessage(
                                            role="user",
                                            content=[TClaudeContentBlock(type="text", text=prompt)],
                                        )
                                    ],
                                ),
                            ).model_dump_json()
                            + "\n"
                        )
            except Exception as e:
                print(f"Error generating batch_extraction_prompt: {e}")
                return False
        return True

    def prepare_batch_glean_prompt(
        self,
        documents: Iterable[Iterable[TChunk]],
        output_path: Path,
        extracted_content: Dict[str, TBedrockBatchOutput],
        prompt_kwargs: Dict[str, str],
        entity_types: List[str],
    ) -> bool:
        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:
            try:
                for document in documents:
                    for chunk in document:
                        global extract_count
                        global extract_error_count
                        extract_count += 1
                        record_id = generate_stable_id(chunk.id)
                        extraction_response = extracted_content.get(record_id)
                        # TODO: log elements that don't pass validation
                        if not extraction_response:
                            print(f"prepare_batch_glean_prompt found no extraction_response for {record_id}")
                            extract_error_count += 1
                            continue

                        messages = extraction_response.modelInput.messages
                        data = extract_json_from_llm_response(extraction_response)
                        # TODO: log elements that don't pass validation
                        if not data:
                            print(f"prepare_batch_glean_prompt found no json for {extraction_response}")
                            extract_error_count += 1
                            continue

                        messages.append(
                            TClaudeMessage(
                                role="assistant",
                                content=[TClaudeContentBlock(type="text", text=json.dumps(data))],
                            )
                        )

                        prompt = format_and_write_prompt(
                            prompt_key="entity_relationship_continue_extraction",
                            format_kwargs={},
                        )

                        messages.append(
                            TClaudeMessage(
                                role="user",
                                content=[TClaudeContentBlock(type="text", text=prompt)],
                            )
                        )

                        f.write(
                            TBedrockBatchRequest(
                                recordId=record_id,
                                modelInput=TBedrockBatchInput(
                                    anthropic_version="bedrock-2023-05-31",
                                    max_tokens=4096,
                                    tool_choice={
                                        "type": "tool",
                                        "name": "extract_entity_relation",
                                    },
                                    tools=TOOLS,
                                    messages=messages,
                                ),
                            ).model_dump_json()
                            + "\n"
                        )
            except Exception as e:
                print(f"Error generating batch_glean_prompt: {e}")
                return False
        print(f"Extraction error rate: {extract_error_count / extract_count}")
        return True

    async def create_graphs(
        self,
        llm: BaseLLMService,
        documents: Iterable[Iterable[TChunk]],
        extracted_content: Dict[str, TBedrockBatchOutput],
        gleaned_content: Dict[str, TBedrockBatchOutput],
        entity_types: List[str],
    ) -> List[asyncio.Future[Optional[BaseGraphStorage[TEntity, TRelation, GTId]]]]:
        _clean_entity_types = [re.sub("[ _]", "", entity_type).upper() for entity_type in entity_types]

        list_of_merge_tasks = []

        for document in documents:
            graphs = []
            for chunk in document:
                global extract_count
                global extract_error_count
                global glean_error_count
                global glean_count
                global glean_error_count
                extract_count += 1
                glean_count += 1
                record_id = generate_stable_id(chunk.id)

                extraction_response = extracted_content.get(record_id)
                if not extraction_response:
                    print(f"create_graphs found no extraction_response for {record_id}")
                    extract_error_count += 1
                    continue

                extraction_data = extract_json_from_llm_response(extraction_response)
                if not extraction_data:
                    print(f"create_graphs found no json for {extraction_response}")
                    extract_error_count += 1
                    continue

                # TODO: log elements that don't pass validation
                entities = [
                    TEntity(name=x["name"], type=x["type"], description=x["desc"])
                    for x in extraction_data["entities"]
                    if all(key in x for key in ["name", "type", "desc"])
                ]
                relationships = []
                extracted_relationships = extraction_data.get("relationships", [])
                extracted_other_relationships = extraction_data.get("other_relationships", [])
                if isinstance(extracted_relationships, list) and isinstance(extracted_other_relationships, list):
                    relationships = [
                        TRelation(
                            source=x["source"],
                            target=x["target"],
                            description=x["desc"],
                        )
                        for x in (extracted_relationships + extracted_other_relationships)
                        if all(key in x for key in ["source", "target", "desc"])
                    ]
                else:
                    extract_error_count += 1
                graph = TGraph(entities=entities, relationships=relationships)

                glean_response = gleaned_content.get(record_id)
                if not glean_response:
                    print(f"create_graphs found no glean_response {record_id}")
                    glean_error_count += 1
                if glean_response:
                    glean_data = extract_json_from_llm_response(glean_response)
                    if not glean_data:
                        print(f"create_graphs found no glean_data json {glean_data}")
                        glean_error_count += 1
                    if glean_data:
                        entities = [
                            TEntity(name=x["name"], type=x["type"], description=x["desc"])
                            for x in glean_data.get("entities", [])
                            if all(key in x for key in ["name", "type", "desc"])
                        ]

                        relationships = []
                        gleaned_relationships = glean_data.get("relationships", [])
                        gleaned_other_relationships = glean_data.get("other_relationships", [])
                        if isinstance(gleaned_relationships, list) and isinstance(gleaned_other_relationships, list):
                            relationships = [
                                TRelation(
                                    source=x["source"],
                                    target=x["target"],
                                    description=x["desc"],
                                )
                                for x in (gleaned_relationships + gleaned_other_relationships)
                                if all(key in x for key in ["source", "target", "desc"])
                            ]
                        else:
                            glean_error_count += 1
                        graph.entities.extend(entities)
                        graph.relationships.extend(relationships)
                for entity in graph.entities:
                    if re.sub("[ _]", "", entity.type).upper() not in _clean_entity_types:
                        entity.type = "UNKNOWN"
                for relationship in graph.relationships:
                    relationship.chunks = [chunk.id]
                graphs.append(graph)
            list_of_merge_tasks.append(asyncio.create_task(self._merge(llm, graphs)))

        print(f"Glean extract error rate: {extract_error_count / extract_count}")
        print(f"Glean error rate: {glean_error_count / glean_count}")
        return list_of_merge_tasks

    async def _merge(self, llm: BaseLLMService, graphs: List[TGraph]) -> BaseGraphStorage[TEntity, TRelation, GTId]:
        """"""
        graph_storage = IGraphStorage[TEntity, TRelation, GTId](config=IGraphStorageConfig(TEntity, TRelation))

        await graph_storage.insert_start()

        try:
            # This is synchronous since each sub graph is inserted into the graph storage and conflicts are resolved
            for graph in graphs:
                await self.graph_upsert(llm, graph_storage, graph.entities, graph.relationships)
        finally:
            await graph_storage.insert_done()

        return graph_storage
