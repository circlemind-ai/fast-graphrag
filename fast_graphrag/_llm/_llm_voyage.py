"""LLM Services module."""
import asyncio
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, List, Optional
import os

import numpy as np
from voyageai import client_async
from voyageai.object.embeddings import EmbeddingsObject

from fast_graphrag._utils import logger

from ._base import BaseEmbeddingService, NoopAsyncContextManager
from aiolimiter import AsyncLimiter

@dataclass
class VoyageAIEmbeddingService(BaseEmbeddingService):
    """Base class for VoyageAI embeddings implementations."""

    embedding_dim: int = field(default=1024)
    max_elements_per_request: int = field(default=128) # Max 128 elements per batch for Voyage API
    model: Optional[str] = field(default="voyage-3")
    api_version: Optional[str] = field(default=None)
    max_requests_concurrent: int = field(default=int(os.getenv("CONCURRENT_TASK_LIMIT", 1024)))
    max_requests_per_minute: int = field(default=1800)
    max_requests_per_second: int = field(default=100)
    rate_limit_per_second: bool = field(default=False)

    def __post_init__(self):
        self.embedding_max_requests_concurrent = (
            asyncio.Semaphore(self.max_requests_concurrent)
            if self.rate_limit_concurrency
            else NoopAsyncContextManager()
        )
        self.embedding_per_minute_limiter = (
            AsyncLimiter(self.max_requests_per_minute, 60)
            if self.rate_limit_per_minute
            else NoopAsyncContextManager()
        )
        self.embedding_per_second_limiter = (
            AsyncLimiter(self.max_requests_per_second, 1)
            if self.rate_limit_per_second
            else NoopAsyncContextManager()
        )
        self.embedding_async_client: client_async.AsyncClient = client_async.AsyncClient(api_key=self.api_key, max_retries=4)
        logger.debug("Initialized VoyageAIEmbeddingService.")

    async def encode(self, texts: list[str], model: Optional[str] = None) -> np.ndarray[Any, np.dtype[np.float32]]:
        try:
            """Get the embedding representation of the input text.

            Args:
                texts (str): The input text to embed.
                model (str, optional): The name of the model to use. Defaults to the model provided in the config.

            Returns:
                list[float]: The embedding vector as a list of floats.
            """
            logger.debug(f"Getting embedding for texts: {texts}")
            model = model or self.model
            if model is None:
                raise ValueError("Model name must be provided.")

            batched_texts = [
                texts[i * self.max_elements_per_request : (i + 1) * self.max_elements_per_request]
                for i in range((len(texts) + self.max_elements_per_request - 1) // self.max_elements_per_request)
            ]
            response = await asyncio.gather(*[self._embedding_request(b, model) for b in batched_texts])

            data = chain(*[r.embeddings for r in response])
            embeddings = np.array([dp for dp in data])
            logger.debug(f"Received embedding response: {len(embeddings)} embeddings")

            return embeddings
        except Exception as e:
            logger.exception("An error occurred:", exc_info=True)
            raise

    async def _embedding_request(self, input: List[str], model: str) -> EmbeddingsObject:
        async with self.embedding_max_requests_concurrent:
            async with self.embedding_per_minute_limiter:
                async with self.embedding_per_second_limiter:
                        return await self.embedding_async_client.embed(model=model, texts=input, output_dimension=self.embedding_dim)
