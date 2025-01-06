"""This module implements a Graph-based Retrieval-Augmented Generation (GraphRAG) system."""

import asyncio
from dataclasses import dataclass, field
import json
from typing import Any, Dict, Generic, Iterable, List, Optional, Tuple, Union
from pathlib import Path
from tqdm import tqdm

import pickle
from fast_graphrag._llm import BaseLLMService, format_and_send_prompt
from fast_graphrag._llm._base import BaseEmbeddingService
from fast_graphrag._models import TAnswer, TBedrockBatchResponse
from fast_graphrag._policies._base import (
    BaseEdgeUpsertPolicy,
    BaseGraphUpsertPolicy,
    BaseNodeUpsertPolicy,
    BatchBaseNodeUpsertPolicy,
    BatchBaseNodeUpsertPromptPolicy,
    BatchBaseEdgeUpsertPolicy,
    BatchBaseEdgeUpsertPromptPolicy,
)
from fast_graphrag._prompt import PROMPTS
from fast_graphrag._services._chunk_extraction import BaseChunkingService
from fast_graphrag._services._information_extraction import (
    BaseInformationExtractionService,
)
from fast_graphrag._services._state_manager import BaseStateManagerService
from fast_graphrag._storage._base import (
    BaseGraphStorage,
    BaseIndexedKeyValueStorage,
    BaseVectorStorage,
)
from fast_graphrag._types import (
    GTChunk,
    GTEdge,
    GTEmbedding,
    GTHash,
    GTId,
    GTNode,
    TChunk,
    TContext,
    TDocument,
    TEntity,
    TId,
    TQueryResponse,
    TRelation,
)
from fast_graphrag._utils import TOKEN_TO_CHAR_RATIO, get_event_loop, logger


@dataclass
class InsertParam:
    pass


@dataclass
class QueryParam:
    with_references: bool = field(default=False)
    only_context: bool = field(default=False)
    entities_max_tokens: int = field(default=4000)
    relations_max_tokens: int = field(default=3000)
    chunks_max_tokens: int = field(default=9000)


@dataclass
class BaseGraphRAG(Generic[GTEmbedding, GTHash, GTChunk, GTNode, GTEdge, GTId]):
    """A class representing a Graph-based Retrieval-Augmented Generation system."""

    working_dir: str = field()
    domain: str = field()
    example_queries: str = field()
    entity_types: List[str] = field()
    n_checkpoints: int = field(default=0)

    llm_service: BaseLLMService = field(init=False, default_factory=lambda: BaseLLMService())
    chunking_service: BaseChunkingService[GTChunk] = field(init=False, default_factory=lambda: BaseChunkingService())
    information_extraction_service: BaseInformationExtractionService[GTChunk, GTNode, GTEdge, GTId] = field(
        init=False,
        default_factory=lambda: BaseInformationExtractionService(
            graph_upsert=BaseGraphUpsertPolicy(
                config=None,
                nodes_upsert_cls=BaseNodeUpsertPolicy,
                edges_upsert_cls=BaseEdgeUpsertPolicy,
            )
        ),
    )
    batch_information_extraction_service: BaseInformationExtractionService[GTChunk, GTNode, GTEdge, GTId] = field(
        init=False,
        default_factory=lambda: BaseInformationExtractionService(
            graph_upsert=BaseGraphUpsertPolicy(
                config=None,
                nodes_upsert_cls=BaseNodeUpsertPolicy,
                edges_upsert_cls=BaseEdgeUpsertPolicy,
            )
        ),
    )
    state_manager: BaseStateManagerService[GTNode, GTEdge, GTHash, GTChunk, GTId, GTEmbedding] = field(
        init=False,
        default_factory=lambda: BaseStateManagerService(
            workspace=None,
            graph_storage=BaseGraphStorage[GTNode, GTEdge, GTId](config=None),
            entity_storage=BaseVectorStorage[GTId, GTEmbedding](config=None),
            chunk_storage=BaseIndexedKeyValueStorage[GTHash, GTChunk](config=None),
            embedding_service=BaseEmbeddingService(),
            node_upsert_policy=BaseNodeUpsertPolicy(config=None),
            batch_node_upsert_policy=BatchBaseNodeUpsertPolicy(config=None),
            batch_node_upsert_prompt_policy=BatchBaseNodeUpsertPromptPolicy(config=None),
            edge_upsert_policy=BaseEdgeUpsertPolicy(config=None),
            batch_edge_upsert_policy=BatchBaseEdgeUpsertPolicy(config=None),
            batch_edge_upsert_prompt_policy=BatchBaseEdgeUpsertPromptPolicy(config=None),
        ),
    )

    def prepare_batch_extraction_prompt(
        self,
        content: Union[str, List[str]],
        dest_file_path: Union[str, Path],
        metadata: Union[List[Optional[Dict[str, Any]]], Optional[Dict[str, Any]]] = None,
        params: Optional[InsertParam] = None,
        show_progress: bool = True,
    ) -> (Iterable[Iterable[TChunk]], str):
        return get_event_loop().run_until_complete(
            self.async_prepare_batch_extraction_prompt(content, dest_file_path, metadata, params, show_progress)
        )

    async def async_prepare_batch_extraction_prompt(
        self,
        content: Union[str, List[str]],
        dest_file_path: Union[str, Path],
        metadata: Union[List[Optional[Dict[str, Any]]], Optional[Dict[str, Any]]] = None,
        params: Optional[InsertParam] = None,
        show_progress: bool = True,
    ) -> (Iterable[Iterable[TChunk]], bool):
        if params is None:
            params = InsertParam()

        if isinstance(content, str):
            content = [content]
        if isinstance(metadata, dict):
            metadata = [metadata]
        if isinstance(dest_file_path, str):
            dest_file_path = Path(dest_file_path)

        if metadata is None or isinstance(metadata, dict):
            data = (TDocument(data=c, metadata=metadata or {}) for c in content)
        else:
            data = (TDocument(data=c, metadata=m or {}) for c, m in zip(content, metadata))

        try:
            await self.state_manager.insert_start()
            # Chunk the data
            chunked_documents = await self.chunking_service.extract(data=data)

            # Filter the chunks checking for duplicates
            new_chunks_per_data = await self.state_manager.filter_new_chunks(chunks_per_data=chunked_documents)
            new_chunks_per_data = chunked_documents

            return (
                new_chunks_per_data,
                self.batch_information_extraction_service.prepare_batch_extraction_prompt(
                    documents=new_chunks_per_data,
                    output_path=dest_file_path,
                    prompt_kwargs={
                        "domain": self.domain,
                        "example_queries": self.example_queries,
                        "entity_types": ",".join(self.entity_types),
                    },
                    entity_types=self.entity_types,
                ),
            )
        except Exception as e:
            logger.error(f"Error during prepare_batch_extraction_prompt: {e}")
            raise e

    def prepare_batch_glean_prompt(
        self,
        chunks: Iterable[Iterable[TChunk]],
        dest_file_path: Union[str, Path],
        extraction_content: Union[str, List[str]],
        metadata: Union[List[Optional[Dict[str, Any]]], Optional[Dict[str, Any]]] = None,
        params: Optional[InsertParam] = None,
        show_progress: bool = True,
    ) -> bool:
        return get_event_loop().run_until_complete(
            self.async_prepare_batch_glean_prompt(
                chunks,
                dest_file_path,
                extraction_content,
                metadata,
                params,
                show_progress,
            )
        )

    async def async_prepare_batch_glean_prompt(
        self,
        chunks: Iterable[Iterable[TChunk]],
        dest_file_path: Union[str, Path],
        extraction_content: Union[str, List[str]],
        metadata: Union[List[Optional[Dict[str, Any]]], Optional[Dict[str, Any]]] = None,
        params: Optional[InsertParam] = None,
        show_progress: bool = True,
    ) -> bool:
        if params is None:
            params = InsertParam()

        if isinstance(metadata, dict):
            metadata = [metadata]
        if isinstance(extraction_content, str):
            extraction_content = [extraction_content]
        if isinstance(dest_file_path, str):
            dest_file_path = Path(dest_file_path)

        extract_prompt_responses = {}
        for line in extraction_content:
            data = json.loads(line.strip())
            record = TBedrockBatchResponse.parse_obj(data)
            extract_prompt_responses[record.recordId] = record

        try:
            await self.state_manager.insert_start()

            return self.batch_information_extraction_service.prepare_batch_glean_prompt(
                documents=chunks,
                output_path=dest_file_path,
                extracted_content=extract_prompt_responses,
                prompt_kwargs={
                    "domain": self.domain,
                    "example_queries": self.example_queries,
                    "entity_types": ",".join(self.entity_types),
                },
                entity_types=self.entity_types,
            )
        except Exception as e:
            logger.error(f"Error during prepare_batch_glean_prompt: {e}")
            raise e

    def batch_insert(
        self,
        chunks: Iterable[Iterable[TChunk]],
        extraction_content: Union[str, List[str]],
        gleaned_content: Union[str, List[str]],
        metadata: Union[List[Optional[Dict[str, Any]]], Optional[Dict[str, Any]]] = None,
        params: Optional[InsertParam] = None,
        show_progress: bool = True,
    ) -> List[asyncio.Future[Optional[BaseGraphStorage[TEntity, TRelation, GTId]]]]:
        return get_event_loop().run_until_complete(
            self.async_batch_insert(chunks, extraction_content, gleaned_content, metadata, show_progress)
        )

    async def async_batch_insert(
        self,
        chunks: Iterable[Iterable[TChunk]],
        extraction_content: Union[str, List[str]],
        gleaned_content: Union[str, List[str]],
        metadata: Union[List[Optional[Dict[str, Any]]], Optional[Dict[str, Any]]] = None,
        params: Optional[InsertParam] = None,
        show_progress: bool = True,
    ) -> List[asyncio.Future[Optional[BaseGraphStorage[TEntity, TRelation, GTId]]]]:
        if params is None:
            params = InsertParam()

        if isinstance(metadata, dict):
            metadata = [metadata]
        if isinstance(extraction_content, str):
            extraction_content = [extraction_content]
        if isinstance(gleaned_content, str):
            gleaned_content = [gleaned_content]

        extract_prompt_responses = {}
        for line in extraction_content:
            data = json.loads(line.strip())
            try:
                record = TBedrockBatchResponse.parse_obj(data)
                extract_prompt_responses[record.recordId] = record
            except Exception as e:
                print(f"Failed to parse extraction response {data}. Exception {e}")

        glean_prompt_responses = {}
        for line in gleaned_content:
            data = json.loads(line.strip())
            try:
                record = TBedrockBatchResponse.parse_obj(data)
                glean_prompt_responses[record.recordId] = record
            except Exception as e:
                print(f"Failed to parse glean response {data}. Exception {e}")

        try:
            subgraphs = await self.batch_information_extraction_service.create_graphs(
                llm=self.llm_service,
                documents=chunks,
                extracted_content=extract_prompt_responses,
                gleaned_content=glean_prompt_responses,
                entity_types=self.entity_types,
            )

            if len(subgraphs) == 0:
                logger.info("No new entities or relationships extracted from the data.")

            return subgraphs
        except Exception as e:
            logger.error(f"Error during async_batch_insertion: {e}")
            raise e

    def batch_generate_graphs(
        self,
        subgraphs: List[asyncio.Future[Optional[BaseGraphStorage[TEntity, TRelation, TId]]]],
        documents: Iterable[Iterable[TChunk]],
        cache_path=Path,
        show_progress: bool = True,
    ) -> Union[bool, None]:
        return get_event_loop().run_until_complete(
            self.async_batch_generate_graphs(subgraphs, documents, cache_path, show_progress)
        )

    async def async_batch_generate_graphs(
        self,
        subgraphs: List[asyncio.Future[Optional[BaseGraphStorage[TEntity, TRelation, TId]]]],
        documents: Iterable[Iterable[TChunk]],
        cache_path=Path,
        show_progress: bool = True,
    ):
        # STEP: Extracting subgraphs
        async def _get_graphs(
            fgraph: asyncio.Future[Optional[BaseGraphStorage[TEntity, TRelation, TId]]],
        ) -> Optional[Tuple[List[TEntity], List[TRelation]]]:
            graph = await fgraph
            if graph is None:
                return None

            nodes = [t for i in range(await graph.node_count()) if (t := await graph.get_node_by_index(i)) is not None]
            edges = [t for i in range(await graph.edge_count()) if (t := await graph.get_edge_by_index(i)) is not None]

            return (nodes, edges)

        graphs = []

        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    graphs = pickle.load(f)
                logger.info(f"Loaded {len(graphs)} graphs from cache: {cache_path}")
            except Exception as e:
                print(f"Failed to load cache from {cache_path}: {e}")
                return
        else:
            graphs = [
                r
                for graph in tqdm(
                    asyncio.as_completed([_get_graphs(fgraph) for fgraph in subgraphs]),
                    total=len(subgraphs),
                    desc="Extracting data",
                    disable=not show_progress,
                )
                if (r := await graph) is not None
            ]
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(graphs, f)
                logger.info(f"Saved {len(graphs)} graphs to cache: {cache_path}")
            except Exception as e:
                print(f"Failed to save cache to {cache_path}: {e}")

        return graphs

    def prepare_batch_node_summaries(self, graphs, summarize_node_prompt_path: Path):
        return get_event_loop().run_until_complete(
            self.async_prepare_batch_node_summaries(graphs, summarize_node_prompt_path)
        )

    async def async_prepare_batch_node_summaries(self, graphs, summarize_node_prompt_path: Path):
        return await self.state_manager.batch_prepare_node_summaries(graphs, summarize_node_prompt_path)

    def prepare_batch_edge_summaries(
        self,
        graphs,
        summarize_nodes_response,
        summarize_edge_prompt_path: Path,
        cache_path: Path,
    ):
        return get_event_loop().run_until_complete(
            self.async_prepare_batch_edge_summaries(
                graphs, summarize_nodes_response, summarize_edge_prompt_path, cache_path
            )
        )

    async def async_prepare_batch_edge_summaries(
        self,
        graphs,
        summarize_nodes_response,
        summarize_edge_prompt_path: Path,
        cache_path: Path,
    ):
        summarize_node_output = {}
        for line in summarize_nodes_response:
            data = json.loads(line.strip())
            record = TBedrockBatchResponse.parse_obj(data)
            summarize_node_output[record.recordId] = record

        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    upserted_nodes = pickle.load(f)
                logger.info(f"Loaded {len(upserted_nodes)} upserted_nodes from cache: {cache_path}")
            except Exception as e:
                print(f"Failed to load cache from {cache_path}: {e}")
                return
        else:
            upserted_nodes = await self.state_manager.batch_upsert_nodes(
                llm=self.llm_service,
                graphs=graphs,
                summarize_node_output=summarize_node_output,
            )
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(upserted_nodes, f)
                logger.info(f"Saved {len(upserted_nodes)} upserted_nodes to cache: {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save cache to {cache_path}: {e}")

        await self.state_manager.batch_prepare_edge_summaries(graphs, summarize_edge_prompt_path)

        return upserted_nodes

    def batch_upsert(
        self,
        chunks: Iterable[Iterable[TChunk]],
        graphs,
        upserted_nodes,
        summarize_edge_responses: Union[str, List[str]],
        metadata: Union[List[Optional[Dict[str, Any]]], Optional[Dict[str, Any]]] = None,
        params: Optional[InsertParam] = None,
        show_progress: bool = True,
    ) -> Tuple[int, int, int]:
        return get_event_loop().run_until_complete(
            self.async_batch_upsert(
                chunks,
                graphs,
                upserted_nodes,
                summarize_edge_responses,
                metadata,
                params,
                show_progress,
            )
        )

    async def async_batch_upsert(
        self,
        chunks: Iterable[Iterable[TChunk]],
        graphs,
        upserted_nodes,
        summarize_edge_response: Union[str, List[str]],
        metadata: Union[List[Optional[Dict[str, Any]]], Optional[Dict[str, Any]]] = None,
        params: Optional[InsertParam] = None,
        show_progress: bool = True,
    ) -> Tuple[int, int, int]:
        if params is None:
            params = InsertParam()
        if isinstance(metadata, dict):
            metadata = [metadata]

        summarize_edge_output = {}
        for line in summarize_edge_response:
            data = json.loads(line.strip())
            try:
                record = TBedrockBatchResponse.parse_obj(data)
                summarize_edge_output[record.recordId] = record
            except Exception:
                print(f"Failed to parse {json.dumps(data, indent=2)}")

        try:
            # Update the graph with the new entities, relationships, and chunks
            await self.state_manager.batch_upsert(
                llm=self.llm_service,
                graphs=graphs,
                documents=chunks,
                upserted_nodes=upserted_nodes,
                summarize_edge_output=summarize_edge_output,
                show_progress=show_progress,
            )

            # Commit the changes if all is successful
            await self.state_manager.insert_done()

            # Return the total number of entities, relationships, and chunks
            return (
                await self.state_manager.get_num_entities(),
                await self.state_manager.get_num_relations(),
                await self.state_manager.get_num_chunks(),
            )
        except Exception as e:
            logger.error(f"Error during async_batch_upsert: {e}")
            raise e

    def insert(
        self,
        content: Union[str, List[str]],
        metadata: Union[List[Optional[Dict[str, Any]]], Optional[Dict[str, Any]]] = None,
        params: Optional[InsertParam] = None,
        show_progress: bool = True,
    ) -> Tuple[int, int, int]:
        try:
            return get_event_loop().run_until_complete(
                self.async_insert(content, metadata, params, show_progress),
            )
        except Exception as e:
            get_event_loop().run_until_complete(self.state_manager.insert_abort())
            raise e

    async def async_insert(
        self,
        content: Union[str, List[str]],
        metadata: Union[List[Optional[Dict[str, Any]]], Optional[Dict[str, Any]]] = None,
        params: Optional[InsertParam] = None,
        show_progress: bool = True,
    ) -> Tuple[int, int, int]:
        """Insert a new memory or memories into the graph.

        Args:
            content (str | list[str]): The data to be inserted. Can be a single string or a list of strings.
            metadata (dict, optional): Additional metadata associated with the data. Defaults to None.
            params (InsertParam, optional): Additional parameters for the insertion. Defaults to None.
            show_progress (bool, optional): Whether to show the progress bar. Defaults to True.
        """
        if params is None:
            params = InsertParam()

        if isinstance(content, str):
            content = [content]
        if isinstance(metadata, dict):
            metadata = [metadata]

        if metadata is None or isinstance(metadata, dict):
            data = (TDocument(data=c, metadata=metadata or {}) for c in content)
        else:
            data = (TDocument(data=c, metadata=m or {}) for c, m in zip(content, metadata))

        await self.state_manager.insert_start()
        # Chunk the data
        chunked_documents = await self.chunking_service.extract(data=data)

        # Filter the chunks checking for duplicates
        new_chunks_per_data = await self.state_manager.filter_new_chunks(chunks_per_data=chunked_documents)
        # Extract entities and relationships from the new chunks only
        subgraphs = self.information_extraction_service.extract(
            llm=self.llm_service,
            documents=new_chunks_per_data,
            prompt_kwargs={
                "domain": self.domain,
                "example_queries": self.example_queries,
                "entity_types": ",".join(self.entity_types),
            },
            entity_types=self.entity_types,
        )
        # Then wait for all of them
        completed_subgraphs = await asyncio.gather(*subgraphs, return_exceptions=True)
        for result in completed_subgraphs:
            if isinstance(result, Exception):
                logger.error(f"Error during chunk extraction: {result}")
                raise result

        # Update the graph with the new entities, relationships, and chunks
        await self.state_manager.upsert(
            llm=self.llm_service,
            subgraphs=completed_subgraphs,
            documents=new_chunks_per_data,
            show_progress=show_progress,
        )

        # Commit the changes if all is successful
        await self.state_manager.insert_done()

        # Return the total number of entities, relationships, and chunks
        return (
            await self.state_manager.get_num_entities(),
            await self.state_manager.get_num_relations(),
            await self.state_manager.get_num_chunks(),
        )

    def query(self, query: str, params: Optional[QueryParam] = None) -> TQueryResponse[GTNode, GTEdge, GTHash, GTChunk]:
        async def _query() -> TQueryResponse[GTNode, GTEdge, GTHash, GTChunk]:
            await self.state_manager.query_start()
            try:
                answer = await self.async_query(query, params)
                return answer
            except Exception as e:
                logger.error(f"Error during query: {e}")
                raise e
            finally:
                await self.state_manager.query_done()

        return get_event_loop().run_until_complete(_query())

    async def async_query(
        self, query: Optional[str], params: Optional[QueryParam] = None
    ) -> TQueryResponse[GTNode, GTEdge, GTHash, GTChunk]:
        """Query the graph with a given input.

        Args:
            query (str): The query string to search for in the graph.
            params (QueryParam, optional): Additional parameters for the query. Defaults to None.

        Returns:
            TQueryResponse: The result of the query (response + context).
        """
        if query is None or len(query) == 0:
            return TQueryResponse[GTNode, GTEdge, GTHash, GTChunk](
                response=PROMPTS["fail_response"], context=TContext([], [], [])
            )
        if params is None:
            params = QueryParam()

        # Extract entities from query
        extracted_entities = await self.information_extraction_service.extract_entities_from_query(
            llm=self.llm_service, query=query, prompt_kwargs={}
        )

        # Retrieve relevant state
        context = await self.state_manager.get_context(query=query, entities=extracted_entities)
        if context is None:
            return TQueryResponse[GTNode, GTEdge, GTHash, GTChunk](
                response=PROMPTS["fail_response"], context=TContext([], [], [])
            )

        # Ask LLM
        context_str = context.truncate(
            max_chars={
                "entities": params.entities_max_tokens * TOKEN_TO_CHAR_RATIO,
                "relations": params.relations_max_tokens * TOKEN_TO_CHAR_RATIO,
                "chunks": params.chunks_max_tokens * TOKEN_TO_CHAR_RATIO,
            },
            output_context_str=not params.only_context,
        )
        if params.only_context:
            answer = ""
        else:
            llm_response, _ = await format_and_send_prompt(
                prompt_key=(
                    "generate_response_query_with_references"
                    if params.with_references
                    else "generate_response_query_no_references"
                ),
                llm=self.llm_service,
                format_kwargs={"query": query, "context": context_str},
                response_model=TAnswer,
            )
            answer = llm_response.answer

        return TQueryResponse[GTNode, GTEdge, GTHash, GTChunk](response=answer, context=context)

    def save_graphml(self, output_path: str) -> None:
        """Save the graph in GraphML format."""

        async def _save_graphml() -> None:
            await self.state_manager.query_start()
            try:
                await self.state_manager.save_graphml(output_path)
            finally:
                await self.state_manager.query_done()

        get_event_loop().run_until_complete(_save_graphml())
