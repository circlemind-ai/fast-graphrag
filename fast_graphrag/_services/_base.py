import asyncio
from dataclasses import dataclass, field
from typing import Dict, Generic, Iterable, List, Optional, Type, Union
from pathlib import Path

from scipy.sparse import csr_matrix

from fast_graphrag._llm import BaseEmbeddingService, BaseLLMService
from fast_graphrag._policies._base import (
    BaseEdgeUpsertPolicy,
    BaseGraphUpsertPolicy,
    BatchBaseEdgeUpsertPolicy,
    BatchBaseEdgeUpsertPromptPolicy,
    BatchBaseNodeUpsertPolicy,
    BatchBaseNodeUpsertPromptPolicy,
    BaseNodeUpsertPolicy,
    BaseRankingPolicy,
)
from fast_graphrag._storage import (
    BaseBlobStorage,
    BaseGraphStorage,
    BaseIndexedKeyValueStorage,
    BaseVectorStorage,
)
from fast_graphrag._storage._namespace import Workspace
from fast_graphrag._types import (
    GTChunk,
    GTEdge,
    GTEmbedding,
    GTHash,
    GTId,
    GTNode,
    TContext,
    TDocument,
    TIndex,
)


@dataclass
class BaseChunkingService(Generic[GTChunk]):
    """Base class for chunk extractor."""

    def __post__init__(self):
        pass

    async def extract(self, data: Iterable[TDocument]) -> Iterable[Iterable[GTChunk]]:
        """Extract unique chunks from the given data."""
        raise NotImplementedError


@dataclass
class BaseInformationExtractionService(Generic[GTChunk, GTNode, GTEdge, GTId]):
    """Base class for entity and relationship extractors."""

    graph_upsert: BaseGraphUpsertPolicy[GTNode, GTEdge, GTId]
    max_gleaning_steps: int = 0

    def extract(
        self,
        llm: BaseLLMService,
        documents: Iterable[Iterable[GTChunk]],
        prompt_kwargs: Dict[str, str],
        entity_types: List[str],
    ) -> List[asyncio.Future[Optional[BaseGraphStorage[GTNode, GTEdge, GTId]]]]:
        """Extract both entities and relationships from the given data."""
        raise NotImplementedError

    async def extract_entities_from_query(
        self, llm: BaseLLMService, query: str, prompt_kwargs: Dict[str, str]
    ) -> Dict[str, List[str]]:
        """Extract entities from the given query."""
        raise NotImplementedError


@dataclass
class BaseStateManagerService(Generic[GTNode, GTEdge, GTHash, GTChunk, GTId, GTEmbedding]):
    """A class for managing state operations."""

    workspace: Optional[Workspace] = field()

    graph_storage: BaseGraphStorage[GTNode, GTEdge, GTId] = field()
    entity_storage: BaseVectorStorage[TIndex, GTEmbedding] = field()
    chunk_storage: BaseIndexedKeyValueStorage[GTHash, GTChunk] = field()

    embedding_service: BaseEmbeddingService = field()

    node_upsert_policy: BaseNodeUpsertPolicy[GTNode, GTId] = field()
    edge_upsert_policy: BaseEdgeUpsertPolicy[GTEdge, GTId] = field()

    batch_node_upsert_policy: BatchBaseNodeUpsertPolicy[GTNode, GTId] = field()
    batch_node_upsert_prompt_policy: BatchBaseNodeUpsertPromptPolicy[GTNode, GTId] = field()
    batch_edge_upsert_policy: BatchBaseEdgeUpsertPolicy[GTEdge, GTId] = field()
    batch_edge_upsert_prompt_policy: BatchBaseEdgeUpsertPromptPolicy[GTEdge, GTId] = field()

    entity_ranking_policy: BaseRankingPolicy = field(default_factory=lambda: BaseRankingPolicy(None))
    relation_ranking_policy: BaseRankingPolicy = field(default_factory=lambda: BaseRankingPolicy(None))
    chunk_ranking_policy: BaseRankingPolicy = field(default_factory=lambda: BaseRankingPolicy(None))

    node_specificity: bool = field(default=False)

    blob_storage_cls: Type[BaseBlobStorage[csr_matrix]] = field(default=BaseBlobStorage)

    async def insert_start(self) -> None:
        """Prepare the storage for indexing before adding new data."""
        raise NotImplementedError

    async def insert_done(self) -> None:
        """Commit the storage operations after indexing."""
        raise NotImplementedError

    async def query_start(self) -> None:
        """Prepare the storage for indexing before adding new data."""
        raise NotImplementedError

    async def query_done(self) -> None:
        """Commit the storage operations after indexing."""
        raise NotImplementedError

    async def filter_new_chunks(self, chunks_per_data: Iterable[Iterable[GTChunk]]) -> List[List[GTChunk]]:
        """Filter the chunks to check for duplicates.

        This method takes a sequence of chunks and returns a sequence of new chunks
        that are not already present in the storage. It uses a hashing mechanism to
        efficiently identify duplicates.

        Args:
            chunks_per_data (Iterable[Iterable[TChunk]]): A sequence of chunks to be filtered.

        Returns:
            Iterable[Iterable[TChunk]]: A sequence of chunks that are not in the storage.
        """
        raise NotImplementedError

    async def upsert(
        self,
        llm: BaseLLMService,
        subgraphs: List[asyncio.Future[Optional[BaseGraphStorage[GTNode, GTEdge, GTId]]]],
        documents: Iterable[Iterable[GTChunk]],
        show_progress: bool = True,
    ) -> None:
        """Clean and upsert entities, relationships, and chunks into the storage."""
        raise NotImplementedError

    async def batch_upsert(
        self,
        llm: BaseLLMService,
        subgraphs: List[asyncio.Future[Optional[BaseGraphStorage[GTNode, GTEdge, GTId]]]],
        documents: Iterable[Iterable[GTChunk]],
        show_progress: bool = True,
        summarize_prompt_path: Optional[Union[str, Path]] = None,
        summarize_responses: Optional[Union[str, List[str]]] = None,
    ) -> None:
        """Clean and upsert entities, relationships, and chunks into the storage."""
        raise NotImplementedError

    async def get_context(
        self, query: str, entities: Dict[str, List[str]]
    ) -> Optional[TContext[GTNode, GTEdge, GTHash, GTChunk]]:
        """Retrieve relevant state from the storage."""
        raise NotImplementedError

    async def get_num_entities(self) -> int:
        """Get the number of entities in the storage."""
        raise NotImplementedError

    async def get_num_relations(self) -> int:
        """Get the number of relations in the storage."""
        raise NotImplementedError

    async def get_num_chunks(self) -> int:
        """Get the number of chunks in the storage."""
        raise NotImplementedError

    async def save_graphml(self, output_path: str) -> None:
        """Save the graph in GraphML format."""
        raise NotImplementedError
