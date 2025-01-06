"""LLM Services module."""

import os
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Tuple, Type, cast

import anthropic
import instructor
import numpy as np
from openai import APIConnectionError, AsyncAzureOpenAI, AsyncOpenAI, RateLimitError
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from fast_graphrag._exceptions import LLMServiceNoResponseError
from fast_graphrag._types import BaseModelAlias
from fast_graphrag._utils import logger, throttle_async_func_call

from ._base import BaseEmbeddingService, BaseLLMService, T_model

TIMEOUT_SECONDS = 600.0


@dataclass
class OpenAILLMService(BaseLLMService):
    """LLM Service for OpenAI LLMs."""

    model: Optional[str] = field(default="gpt-4o-mini")
    mode: instructor.Mode = field(default=instructor.Mode.JSON)
    client: Literal["openai", "anthropic", "azure"] = field(default="openai")
    api_version: Optional[str] = field(default=None)

    def __post_init__(self):
        if self.client == "anthropic":
            self.llm_async_client = instructor.from_anthropic(
                anthropic.AsyncAnthropic(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    timeout=TIMEOUT_SECONDS,
                ),
                mode=instructor.Mode.ANTHROPIC_TOOLS,
            )
        elif self.client == "azure":
            assert self.base_url is not None and self.api_version is not None, (
                "Azure OpenAI requires a base url and an api version."
            )
            self.llm_async_client = instructor.from_openai(
                AsyncAzureOpenAI(
                    azure_endpoint=self.base_url,
                    api_key=self.api_key,
                    api_version=self.api_version,
                    timeout=TIMEOUT_SECONDS,
                ),
                mode=self.mode,
            )
        elif self.client == "openai":
            self.llm_async_client = instructor.from_openai(
                AsyncOpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    timeout=TIMEOUT_SECONDS,
                ),
                mode=self.mode,
            )
        else:
            raise ValueError("Invalid client type. Must be 'openai' or 'azure'")
        logger.debug("Initialized OpenAILLMService with patched OpenAI client.")

    @throttle_async_func_call(
        max_concurrent=int(os.getenv("CONCURRENT_TASK_LIMIT", 1024)),
        stagger_time=0.001,
        waiting_time=0.001,
    )
    async def send_message(
        self,
        prompt: str,
        model: str | None = None,
        system_prompt: str | None = None,
        history_messages: list[dict[str, str]] | None = None,
        response_model: Type[T_model] | None = None,
        **kwargs: Any,
    ) -> Tuple[T_model, list[dict[str, str]]]:
        """Send a message to the language model and receive a response.

        Args:
            prompt (str): The input message to send to the language model.
            model (str): The name of the model to use. Defaults to the model provided in the config.
            system_prompt (str, optional): The system prompt to set the context for the conversation. Defaults to None.
            history_messages (list, optional): A list of previous messages in the conversation. Defaults to empty.
            response_model (Type[T], optional): The Pydantic model to parse the response. Defaults to None.
            **kwargs: Additional keyword arguments that may be required by specific LLM implementations.

        Returns:
            str: The response from the language model.
        """
        logger.debug(f"Sending message with prompt: {prompt}")
        model = model or self.model
        if model is None:
            raise ValueError("Model name must be provided.")
        messages: list[dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            logger.debug(f"Added system prompt: {system_prompt}")

        if history_messages:
            messages.extend(history_messages)
            logger.debug(f"Added history messages: {history_messages}")

        messages.append({"role": "user", "content": prompt})

        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = 8192

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=4, max=10),
                retry=retry_if_exception_type((RateLimitError, TimeoutError)),
            ):
                with attempt:
                    llm_response: T_model = await self.llm_async_client.chat.completions.create(
                        model=model,
                        messages=messages,  # type: ignore
                        response_model=(
                            response_model.Model
                            if response_model and issubclass(response_model, BaseModelAlias)
                            else response_model
                        ),
                        **kwargs,
                    )
        except APIConnectionError as e:
            logger.error(f"Connection to LLM service failed: {e}")
            raise e

        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            raise e

        if not llm_response:
            logger.error("No response received from the language model.")
            raise LLMServiceNoResponseError("No response received from the language model.")

        messages.append(
            {
                "role": "assistant",
                "content": (
                    llm_response.model_dump_json() if isinstance(llm_response, BaseModel) else str(llm_response)
                ),
            }
        )
        logger.debug(f"Received response: {llm_response}")

        if response_model and issubclass(response_model, BaseModelAlias):
            llm_response = cast(
                T_model,
                cast(BaseModelAlias.Model, llm_response).to_dataclass(llm_response),
            )

        return llm_response, messages


@dataclass
class OpenAIEmbeddingService(BaseEmbeddingService):
    """Base class for Language Model implementations."""

    embedding_dim: int = field(default=1536)
    max_elements_per_request: int = field(default=32)
    model: Optional[str] = field(default="text-embedding-3-small")
    client: Literal["openai", "azure"] = field(default="openai")
    api_version: Optional[str] = field(default=None)

    def __post_init__(self):
        if self.client == "azure":
            assert self.base_url is not None and self.api_version is not None, (
                "Azure OpenAI requires a base url and an api version."
            )
            self.embedding_async_client = instructor.from_openai(
                AsyncAzureOpenAI(
                    azure_endpoint=self.base_url,
                    api_key=self.api_key,
                    api_version=self.api_version,
                )
            )
        elif self.client == "openai":
            self.embedding_async_client = instructor.from_openai(
                AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
            )
        else:
            raise ValueError("Invalid client type. Must be 'openai' or 'azure'")
        logger.debug("Initialized OpenAIEmbeddingService with OpenAI client.")

    async def encode(self, texts: list[str], model: Optional[str] = None) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Get the embedding representation of the input text.

        Args:
            texts (str): The input text to embed.
            model (str, optional): The name of the model to use. Defaults to the model provided in the config.

        Returns:
            list[float]: The embedding vector as a list of floats.
        """
        logger.debug(f"Processing embeddings for {len(texts)} texts")
        model = model or self.model
        if model is None:
            raise ValueError("Model name must be provided.")

        all_embeddings = []
        # TODO: put tqdm on this
        for i in range(0, len(texts), self.max_elements_per_request):
            batch = texts[i : i + self.max_elements_per_request]
            response = await self._embedding_request(batch, model)
            batch_embeddings = [dp.embedding for dp in response.data]
            all_embeddings.extend(batch_embeddings)
            logger.debug(
                f"Processed batch {i // self.max_elements_per_request + 1}, total embeddings so far: {len(all_embeddings)}"
            )

        return np.array(all_embeddings)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, TimeoutError)),
        reraise=True,
    )
    async def _embedding_request(self, input: List[str], model: str) -> Any:
        return await self.embedding_async_client.embeddings.create(model=model, input=input, encoding_format="float")
