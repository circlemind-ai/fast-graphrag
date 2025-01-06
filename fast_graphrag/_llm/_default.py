__all__ = ["DefaultLLMService", "DefaultEmbeddingService"]

from ._llm_openai import OpenAIEmbeddingService, OpenAILLMService


class DefaultLLMService(OpenAILLMService):
    pass


class DefaultEmbeddingService(OpenAIEmbeddingService):
    pass
