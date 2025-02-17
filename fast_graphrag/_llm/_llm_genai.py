"""Google Genai LLM Services module for Gemini and Vertex AI endpoints.

This module provides two main services:
• GeminiLLMService - for sending messages/requests to the language model endpoint.
• GeminiEmbeddingService - for obtaining text embeddings via the language model.

Both classes support asynchronous calls, internal retry handling using tenacity,
and rate limiting via semaphores and AsyncLimiter.
"""

import asyncio
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, cast, Literal, Callable, Awaitable
import instructor
from pydantic import BaseModel, ValidationError, TypeAdapter
import requests  # used to catch requests.exceptions.ConnectionError
from google import genai

from fast_graphrag._exceptions import LLMServiceNoResponseError
from fast_graphrag._llm._base import BaseLLMService, BaseEmbeddingService, T_model, NoopAsyncContextManager
from fast_graphrag._models import BaseModelAlias
from aiolimiter import AsyncLimiter
from itertools import chain
from fast_graphrag._utils import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from google.genai import types, errors
from vertexai.preview.tokenization import get_tokenizer_for_model
from json_repair import repair_json


def default_safety_settings() -> List[types.SafetySetting]:
    return [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
    ]
    
# Helper function to execute an asynchronous operation with inner retry logic.
async def _execute_with_inner_retries(
    operation: Callable[[], Awaitable[Any]],
    validate: Callable[[Any, int, int], bool],
    max_attempts: int = 4,
    short_sleep: float = 0.01,
    error_sleep: float = 0.2,
) -> Any:
    """
    Executes an asynchronous operation with a specified number of inner retry attempts.

    Args:
        operation: An async callable representing the API call.
        validate: A function that takes (result, current_attempt, total_attempts)
                  and returns True when the result meets the criteria.
        max_attempts: Total number of inner attempts.
        short_sleep: Sleep interval for minor issues.
        error_sleep: Sleep interval for connection-related issues.

    Returns:
        The valid result returned by operation.

    Raises:
        The last encountered exception if the maximum number of attempts is exceeded.
    """
    last_exception: Exception = Exception("Unknown error")
    for attempt in range(max_attempts):
        try:
            result = await operation()
            if validate(result, attempt, max_attempts):
                return result
        except (errors.ClientError, ConnectionResetError, requests.exceptions.ConnectionError) as e:
            last_exception = e
        except Exception as e:
            last_exception = e
        # Delay before next attempt; use different sleep if it was a connection error.
        await asyncio.sleep(
            error_sleep if isinstance(last_exception, (ConnectionResetError, requests.exceptions.ConnectionError))
            else short_sleep
        )
    raise last_exception

@dataclass
class GeminiLLMService(BaseLLMService):
    # Core fields required to interact with the LLM endpoint.
    model: Optional[str] = field(default="gemini-2.0-flash")
    mode: instructor.Mode = field(default=instructor.Mode.JSON)
    client: Literal["gemini", "vertex"] = field(default="gemini")
    api_key: Optional[str] = field(default=None)
    temperature: float = field(default=0.7)
    candidate_count: int = field(default=1)
    max_requests_concurrent: int = field(default=int(os.getenv("CONCURRENT_TASK_LIMIT", 1024)))
    max_requests_per_minute: int = field(default=2000) # Gemini Flash 2.0 has a paid developer API limit of 2000 RPM
    max_requests_per_second: int = field(default=500)
    project_id: Optional[str] = field(default=None)
    location: Optional[str] = field(default=None)
    safety_settings: list[types.SafetySetting] = field(default_factory=default_safety_settings)

    def __post_init__(self):
        """
        Post-initialization:
        • Sets up concurrency semaphores and rate limiters based on the provided configuration.
        • Instantiates the appropriate asynchronous LLM client for either Vertex or Gemini.
        • Initializes a local tokenizer for the Gemini model.
        """
        self.llm_max_requests_concurrent = (
            asyncio.Semaphore(self.max_requests_concurrent)
            if self.rate_limit_concurrency
            else NoopAsyncContextManager()
        )
        self.llm_per_minute_limiter = (
            AsyncLimiter(self.max_requests_per_minute, 60)
            if self.rate_limit_per_minute
            else NoopAsyncContextManager()
        )
        self.llm_per_second_limiter = (
            AsyncLimiter(self.max_requests_per_second, 1)
            if self.rate_limit_per_second
            else NoopAsyncContextManager()
        )

        # Instantiate the appropriate client based on the provided "client" value.
        if self.client == "vertex":
            # Ensure that either project_id and location are provided or an express API key is used.
            assert (
                (self.project_id is not None and self.location is not None and self.api_key is None)
                or (self.project_id is None and self.location is None and self.api_key is not None)
            ), "Azure OpenAI requires a project id and location, or an express API key."
            if self.api_key is not None:
                self.llm_async_client: genai.Client = genai.Client(vertexai=True, api_key=self.api_key)
            else:
                self.llm_async_client = genai.Client(vertexai=True, project=self.project_id, location=self.location)
        elif self.client == "gemini":
            self.llm_async_client: genai.Client = genai.Client(api_key=self.api_key)
        else:
            raise ValueError("Invalid client type. Must be 'openai' or 'azure'")

        # Initialize local tokenizer for Gemini. Update model name if needed.
        self.tokenizer = get_tokenizer_for_model('gemini-1.5-flash-002')
        logger.debug("Initialized GeminiLLMService.")

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the provided text utilizing the local Gemini tokenizer.
        
        Args:
            text (str): The input text whose tokens are to be counted.
            model (Optional[str]): An optional model override (not used in current implementation).
            **kwargs: Additional keyword arguments (currently unused).
        
        Returns:
            int: Total token count.
        """
        return self.tokenizer.count_tokens(contents=text)

    @retry(
        stop=stop_after_attempt(8),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((TimeoutError, Exception)),
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
        """
        Sends a message to the Gemini AI language model and handles:
          • Concurrency and rate limiting.
          • Request retries with inner attempt loops.
          • Response validation/parsing including JSON repair.
        
        Args:
            prompt (str): The main user input.
            model (Optional[str]): Optional override for the model name.
            system_prompt (Optional[str]): Optional system-level instructions.
            history_messages (Optional[list[dict[str, str]]]): Prior conversation messages.
            response_model (Optional[Type[T_model]]): Pydantic model (or alias) dictating the response structure.
            temperature (float): Generation temperature setting.
            **kwargs: Additional generation parameters (unused here).
        
        Returns:
            Tuple[T_model, list[dict[str, str]]]: A tuple containing the parsed response and the updated message history.
        
        Raises:
            ValueError: If the model name is missing.
            LLMServiceNoResponseError: If no valid response is obtained after all retries.
            errors.APIError: For unrecoverable API errors.
        """
        # Apply the concurrency and rate limiters.
        async with self.llm_max_requests_concurrent:
            async with self.llm_per_minute_limiter:
                async with self.llm_per_second_limiter:
                    # Determine the model to use.
                    model = model or self.model
                    if model is None:
                        raise ValueError("Model name must be provided.")

                    # Build message history including the current user prompt.
                    messages: List[Dict[str, str]] = []
                    if history_messages:
                        messages.extend(history_messages)
                    messages.append({"role": "user", "content": prompt})
                    combined_prompt = "\n".join(
                        [f"{msg['role']}: {msg['content']}" for msg in messages]
                    )

                    try:
                        def validate_generate_content(response: Any, attempt: int, max_attempts: int) -> bool:
                            # Require that a response exists and that "text" is non-empty.
                            if not response or not getattr(response, "text", ""):
                                return False
                            # If a response model is expected, then require a parsed response on nonfinal attempts.
                            if response_model is not None and attempt != (max_attempts - 1):
                                if not getattr(response, "parsed", ""):
                                    return False
                            return True
                        
                        # Configure generation call with safety settings, candidate count, and temperature.
                        generate_config = (
                            types.GenerateContentConfig(
                                system_instruction=system_prompt,
                                response_mime_type="application/json",
                                response_schema=(
                                    (response_model.Model)
                                    if issubclass(response_model, BaseModelAlias)
                                    else (response_model)
                                ),
                                candidate_count=self.candidate_count,
                                temperature=self.temperature,
                                safety_settings=self.safety_settings,
                            )
                            if response_model
                            else types.GenerateContentConfig(
                                system_instruction=system_prompt,
                                candidate_count=self.candidate_count,
                                temperature=self.temperature,
                                safety_settings=self.safety_settings,
                            )
                        )

                        # Use the helper to perform the inner retries.
                        response = await _execute_with_inner_retries(
                            operation=lambda: self.llm_async_client.aio.models.generate_content(
                                model=model,
                                contents=combined_prompt,
                                config=generate_config,
                            ),
                            validate=validate_generate_content,
                            max_attempts=4,
                            short_sleep=0.01,
                            error_sleep=0.2,
                        )

                        if not response or not getattr(response, "text", ""):
                            raise LLMServiceNoResponseError("Failed to obtain a valid response for content.")

                        # Parse and validate the response.
                        try:
                            if response_model:
                                if getattr(response, "parsed"):
                                    if issubclass(response_model, BaseModelAlias):
                                        llm_response = TypeAdapter(response_model.Model).validate_python(getattr(response, "parsed"))
                                    else:
                                        llm_response = TypeAdapter(response_model).validate_python(getattr(response, "parsed"))
                                else:
                                    # Attempt to repair JSON if initial parsing failed.
                                    fixed_json = repair_json(getattr(response, "text"))
                                    if issubclass(response_model, BaseModelAlias):
                                        llm_response = TypeAdapter(response_model.Model).validate_json(fixed_json)
                                    else:
                                        llm_response = TypeAdapter(response_model).validate_json(fixed_json)
                            else:
                                llm_response = response.text
                        except ValidationError as e:
                            raise LLMServiceNoResponseError(f"Invalid JSON response: {str(e)}") from e

                        # Append the AI model's response to the conversation history.
                        messages.append(
                            {
                                "role": "model",
                                "content": (
                                    llm_response.model_dump_json()
                                    if isinstance(llm_response, BaseModel)
                                    else str(llm_response)
                                ),
                            }
                        )

                        # If working with a BaseModelAlias, convert back to the dataclass.
                        if response_model and issubclass(response_model, BaseModelAlias):
                            llm_response = cast(
                                T_model,
                                cast(BaseModelAlias.Model, llm_response).to_dataclass(llm_response),
                            )

                        return llm_response, messages

                    except errors.APIError as e:
                        # Handle API error responses:
                        # • Rate limit errors (HTTP 429): let the retry mechanism take over.
                        # • Other client errors (HTTP 400, 403, 404): log and raise immediately.
                        # • Server errors (HTTP 500, 503, 504): log and raise to allow retry after a delay.
                        if e.code == 429 or (e.details and e.details.get("code") == 429):
                            logger.warning(f"Rate limit error encountered: {e.code} - {e.message}. Attempting retry.")
                            raise
                        elif e.code in (400, 403, 404):
                            logger.error(f"Client error encountered: {e.code} - {e.message}. Check your request parameters or API key.")
                            raise
                        elif e.code in (500, 503, 504):
                            logger.error(f"Server error encountered: {e.code} - {e.message}. Consider retrying after a short delay.")
                            raise
                        else:
                            logger.exception(f"Unexpected API error encountered: {e.code} - {e.message}")
                            raise
                    except Exception as e:
                        logger.exception(f"Unexpected error: {e}")
                        raise


@dataclass
class GeminiEmbeddingService(BaseEmbeddingService):
    """Service implementation to retrieve embeddings for texts using the Gemini model."""
    embedding_dim: int = field(default=768)
    max_elements_per_request: int = field(default=99) # Maximum of 100 elements per batch with Gemini
    model: Optional[str] = field(default="text-embedding-004")
    api_version: Optional[str] = field(default=None)
    max_requests_concurrent: int = field(default=int(os.getenv("CONCURRENT_TASK_LIMIT", 150)))
    max_requests_per_minute: int = field(default=80) # Google cloud enforces strict limits on batch requests to Gemini embedding endpoints
    max_requests_per_second: int = field(default=20)
    
    def __post_init__(self):
        """
        Post-initialization:
        • Sets up concurrency semaphores and rate limiters for embedding requests.
        • Instantiates the asynchronous client for embedding requests.
        """
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
        self.embedding_async_client: genai.Client = genai.Client(api_key=self.api_key)
        logger.debug("Initialized GeminiEmbeddingService.")

    async def encode(self, texts: list[str], model: Optional[str] = None) -> np.ndarray[Any, np.dtype[np.float32]]:
        """
        Obtain embedding vectors for provided input texts.
        
        This method internally splits the texts into batches (based on max_elements_per_request)
        and sends concurrent requests for embedding. The responses are then concatenated into a single numpy array.
        
        Args:
            texts (list[str]): List of input texts to be embedded.
            model (Optional[str]): Optional model override; defaults to the service's model if not provided.
        
        Returns:
            np.ndarray: Array of embedding vectors.
        
        Raises:
            Exception: Propagates any exception encountered during embedding requests.
        """
        try:
            logger.debug(f"Getting embedding for texts: {texts}")
            model = model or self.model
            if model is None:
                raise ValueError("Model name must be provided.")

            # Batch the texts to not exceed the maximum allowed elements per request.
            batched_texts = [
                texts[i * self.max_elements_per_request : (i + 1) * self.max_elements_per_request]
                for i in range((len(texts) + self.max_elements_per_request - 1) // self.max_elements_per_request)
            ]
            # Execute embedding requests concurrently for all batches.
            response = await asyncio.gather(*[self._embedding_request(batch, model) for batch in batched_texts])

            # Flatten the list of responses and create the embedding numpy array.
            data = chain(*[r for r in response])
            embeddings = np.array([dp.values for dp in data])
            logger.debug(f"Received embedding response: {len(embeddings)} embeddings")

            return embeddings
        except Exception as e:
            logger.exception("An error occurred during embedding encoding:", exc_info=True)
            raise

    @retry(
        stop=stop_after_attempt(8),
        wait=wait_exponential(multiplier=1, min=5, max=60),
        retry=retry_if_exception_type((TimeoutError, Exception)),
    )
    async def _embedding_request(self, input: list[Any], model: str) -> list[types.ContentEmbedding]:
        """
        Makes an embedding request for a batch of input texts.

        Applies internal retry logic and rate limiting to ensure a valid response is obtained.
        
        Args:
            input (list[Any]): A batch of texts to be embedded.
            model (str): The model name to be used for generating embeddings.
        
        Returns:
            list[types.ContentEmbedding]: A list of embedding objects.
        
        Raises:
            LLMServiceNoResponseError: If a valid response is not obtained after retries.
            errors.APIError: For unrecoverable API errors.
        """
        async with self.embedding_max_requests_concurrent:
            async with self.embedding_per_minute_limiter:
                async with self.embedding_per_second_limiter:
                    try:
                        def validate_embedding_response(response: Any, attempt: int, max_attempts: int) -> bool:
                            if not response or not getattr(response, "embeddings", None) or getattr(response, "embeddings") == []:
                                return False
                            return True

                        response = await _execute_with_inner_retries(
                            operation=lambda: self.embedding_async_client.aio.models.embed_content(
                                model=model, contents=input
                            ),
                            validate=validate_embedding_response,
                            max_attempts=4,
                            short_sleep=0.01,
                            error_sleep=0.2,
                        )

                        if not response or not getattr(response, "embeddings", None) or getattr(response, "embeddings") == []:
                            raise LLMServiceNoResponseError("Failed to obtain a valid response for embeddings.")

                        return getattr(response, "embeddings")

                    except errors.APIError as e:
                        # Handle various API error scenarios.
                        if e.code == 429 or (e.details and e.details.get("code") == 429):
                            logger.warning(f"Rate limit error encountered: {e.code} - {e.message}. Delegating to outer retry.")
                            raise
                        elif e.code in (400, 403, 404):
                            logger.error(f"Client error encountered: {e.code} - {e.message}. Check your request parameters or API key.")
                            raise
                        elif e.code in (500, 503, 504):
                            logger.error(f"Server error encountered: {e.code} - {e.message}. Consider retrying after a short delay.")
                            raise
                        else:
                            logger.exception(f"Unexpected API error encountered: {e.code} - {e.message}")
                            raise
                    except Exception as e:
                        logger.exception(f"Unexpected error during embedding request: {e}")
                        raise