"""Top-level package for GraphRAG."""

__all__ = ["GraphRAG", "QueryParam"]

from dataclasses import dataclass, field
from typing import Type
import logging

from fast_graphrag._llm import DefaultEmbeddingService, DefaultLLMService
from fast_graphrag._llm._base import BaseEmbeddingService
from fast_graphrag._llm._llm_openai import BaseLLMService
from fast_graphrag._policies._base import BaseGraphUpsertPolicy
from fast_graphrag._policies._batch_graph_upsert import (
    EdgeUpsertPolicy_UpsertValidAndMergeSimilarByLLM_BatchAsync,
    EdgeUpsertPolicy_UpsertValidAndMergeSimilarByLLM_PreparePrompts,
    NodeUpsertPolicy_SummarizeDescription_BatchAsync,
    NodeUpsertPolicy_SummarizeDescription_PreparePrompts,
)
from fast_graphrag._policies._graph_upsert import (
    DefaultGraphUpsertPolicy,
    EdgeUpsertPolicy_UpsertIfValidNodes,
    EdgeUpsertPolicy_UpsertValidAndMergeSimilarByLLM,
    NodeUpsertPolicy_SummarizeDescription,
)
from fast_graphrag._policies._ranking import (
    RankingPolicy_TopK,
    RankingPolicy_WithThreshold,
)
from fast_graphrag._services import (
    BaseChunkingService,
    BaseInformationExtractionService,
    BaseStateManagerService,
    DefaultChunkingService,
    DefaultInformationExtractionService,
    BatchInformationExtractionService,
    DefaultStateManagerService,
)
from fast_graphrag._storage import (
    DefaultGraphStorage,
    DefaultGraphStorageConfig,
    DefaultIndexedKeyValueStorage,
    DefaultVectorStorage,
    DefaultVectorStorageConfig,
)
from fast_graphrag._storage._namespace import Workspace
from fast_graphrag._types import (
    TChunk,
    TEmbedding,
    TEntity,
    THash,
    TId,
    TIndex,
    TRelation,
)

from ._graphrag import BaseGraphRAG, QueryParam


@dataclass
class GraphRAG(BaseGraphRAG[TEmbedding, THash, TChunk, TEntity, TRelation, TId]):
    """A class representing a Graph-based Retrieval-Augmented Generation system."""

    @dataclass
    class Config:
        """Configuration for the GraphRAG class."""

        chunking_service_cls: Type[BaseChunkingService[TChunk]] = field(default=DefaultChunkingService)
        information_extraction_service_cls: Type[BaseInformationExtractionService[TChunk, TEntity, TRelation, TId]] = (
            field(default=DefaultInformationExtractionService)
        )
        batch_information_extraction_service_cls: Type[
            BaseInformationExtractionService[TChunk, TEntity, TRelation, TId]
        ] = field(default=BatchInformationExtractionService)
        information_extraction_upsert_policy: BaseGraphUpsertPolicy[TEntity, TRelation, TId] = field(
            default_factory=lambda: DefaultGraphUpsertPolicy(
                config=NodeUpsertPolicy_SummarizeDescription.Config(),
                nodes_upsert_cls=NodeUpsertPolicy_SummarizeDescription,
                edges_upsert_cls=EdgeUpsertPolicy_UpsertIfValidNodes,
            )
        )
        state_manager_cls: Type[BaseStateManagerService[TEntity, TRelation, THash, TChunk, TId, TEmbedding]] = field(
            default=DefaultStateManagerService
        )

        llm_service: BaseLLMService = field(default_factory=lambda: DefaultLLMService())
        embedding_service: BaseEmbeddingService = field(default_factory=lambda: DefaultEmbeddingService())

        graph_storage: DefaultGraphStorage[TEntity, TRelation, TId] = field(
            default_factory=lambda: DefaultGraphStorage(DefaultGraphStorageConfig(node_cls=TEntity, edge_cls=TRelation))
        )
        entity_storage: DefaultVectorStorage[TIndex, TEmbedding] = field(
            default_factory=lambda: DefaultVectorStorage(DefaultVectorStorageConfig())
        )
        chunk_storage: DefaultIndexedKeyValueStorage[THash, TChunk] = field(
            default_factory=lambda: DefaultIndexedKeyValueStorage(None)
        )

        entity_ranking_policy: RankingPolicy_WithThreshold = field(
            default_factory=lambda: RankingPolicy_WithThreshold(RankingPolicy_WithThreshold.Config(threshold=0.005))
        )
        relation_ranking_policy: RankingPolicy_TopK = field(
            default_factory=lambda: RankingPolicy_TopK(RankingPolicy_TopK.Config(top_k=64))
        )
        chunk_ranking_policy: RankingPolicy_TopK = field(
            default_factory=lambda: RankingPolicy_TopK(RankingPolicy_TopK.Config(top_k=8))
        )
        node_upsert_policy: NodeUpsertPolicy_SummarizeDescription = field(
            default_factory=lambda: NodeUpsertPolicy_SummarizeDescription()
        )
        batch_node_upsert_policy: NodeUpsertPolicy_SummarizeDescription_BatchAsync = field(
            default_factory=lambda: NodeUpsertPolicy_SummarizeDescription_BatchAsync()
        )
        batch_node_upsert_prompt_policy: NodeUpsertPolicy_SummarizeDescription_PreparePrompts() = field(
            default_factory=lambda: NodeUpsertPolicy_SummarizeDescription_PreparePrompts()
        )
        edge_upsert_policy: EdgeUpsertPolicy_UpsertValidAndMergeSimilarByLLM = field(
            default_factory=lambda: EdgeUpsertPolicy_UpsertValidAndMergeSimilarByLLM()
        )
        batch_edge_upsert_policy: EdgeUpsertPolicy_UpsertValidAndMergeSimilarByLLM_BatchAsync = field(
            default_factory=lambda: EdgeUpsertPolicy_UpsertValidAndMergeSimilarByLLM_BatchAsync()
        )
        batch_edge_upsert_prompt_policy: EdgeUpsertPolicy_UpsertValidAndMergeSimilarByLLM_PreparePrompts = field(
            default_factory=lambda: EdgeUpsertPolicy_UpsertValidAndMergeSimilarByLLM_PreparePrompts()
        )

        def __post_init__(self):
            """Initialize the GraphRAG Config class."""
            self.entity_storage.embedding_dim = self.embedding_service.embedding_dim

    config: Config = field(default_factory=Config)

    def __post_init__(self):
        """Initialize the GraphRAG class."""
        self.llm_service = self.config.llm_service
        self.embedding_service = self.config.embedding_service
        self.chunking_service = self.config.chunking_service_cls()
        self.information_extraction_service = self.config.information_extraction_service_cls(
            graph_upsert=self.config.information_extraction_upsert_policy
        )
        self.batch_information_extraction_service = self.config.batch_information_extraction_service_cls(
            graph_upsert=self.config.information_extraction_upsert_policy
        )
        self.state_manager = self.config.state_manager_cls(
            workspace=Workspace.new(self.working_dir, keep_n=self.n_checkpoints),
            embedding_service=self.embedding_service,
            graph_storage=self.config.graph_storage,
            entity_storage=self.config.entity_storage,
            chunk_storage=self.config.chunk_storage,
            entity_ranking_policy=self.config.entity_ranking_policy,
            chunk_ranking_policy=self.config.chunk_ranking_policy,
            node_upsert_policy=self.config.node_upsert_policy,
            batch_node_upsert_policy=self.config.batch_node_upsert_policy,
            batch_node_upsert_prompt_policy=self.config.batch_node_upsert_prompt_policy,
            edge_upsert_policy=self.config.edge_upsert_policy,
            batch_edge_upsert_policy=self.config.batch_edge_upsert_policy,
            batch_edge_upsert_prompt_policy=self.config.batch_edge_upsert_prompt_policy,
        )
        logging.basicConfig(filename="graphrag2.log", level=logging.DEBUG)
