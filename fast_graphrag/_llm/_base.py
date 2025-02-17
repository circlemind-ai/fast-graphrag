"""LLM Services module."""
import os
import re

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Type, TypeVar, Union

import numpy as np
from pydantic import BaseModel

from fast_graphrag._models import BaseModelAlias
from fast_graphrag._prompt import PROMPTS

T_model = TypeVar("T_model", bound=Union[BaseModel, BaseModelAlias])
TOKEN_PATTERN = re.compile(r'\w+|[^\w\s]', re.UNICODE)


async def format_and_send_prompt(
    prompt_key: str,
    llm: "BaseLLMService",
    format_kwargs: dict[str, Any],
    response_model: Type[T_model],
    **args: Any,
) -> Tuple[T_model, list[dict[str, str]]]:
    """
    Get a prompt, format it with the supplied args, and send it to the LLM.

    If a system prompt is provided (i.e. PROMPTS contains a key named
    '{prompt_key}_system'), it will use both the system and prompt entries:
        - System prompt: PROMPTS[prompt_key + '_system']
        - Message prompt: PROMPTS[prompt_key + '_prompt']

    Otherwise, it will default to using the single prompt defined by:
        - PROMPTS[prompt_key]

    Args:
        prompt_key (str): The key for the prompt in the PROMPTS dictionary.
        llm (BaseLLMService): The LLM service to use for sending the message.
        response_model (Type[T_model]): The expected response model.
        format_kwargs (dict[str, Any]): Dictionary of arguments to format the prompt.
        model (str | None): The model to use for the LLM. Defaults to None.
        max_tokens (int | None): The maximum number of tokens for the response. Defaults to None.
        **args (Any): Additional keyword arguments to pass to the LLM.

    Returns:
        Tuple[T_model, list[dict[str, str]]]: The response from the LLM.
    """
    system_key = prompt_key + '_system'

    if system_key in PROMPTS:
        # Use separate system and prompt entries
        system = PROMPTS[system_key]
        prompt = PROMPTS[prompt_key + '_prompt']
        formatted_system = system.format(**format_kwargs)
        formatted_prompt = prompt.format(**format_kwargs)
        return await llm.send_message(system_prompt=formatted_system, prompt=formatted_prompt, response_model=response_model, **args)
    else:
        # Default: use the single prompt entry
        prompt = PROMPTS[prompt_key]
        formatted_prompt = prompt.format(**format_kwargs)
        return await llm.send_message(prompt=formatted_prompt, response_model=response_model, **args)



@dataclass
class BaseLLMService:
    """Base class for Language Model implementations."""

    model: Optional[str] = field(default=None)
    base_url: Optional[str] = field(default=None)
    api_key: Optional[str] = field(default=None)
    llm_async_client: Any = field(init=False, default=None)
    max_requests_concurrent: int = field(default=int(os.getenv("CONCURRENT_TASK_LIMIT", 1024)))
    max_requests_per_minute: int = field(default=500)
    max_requests_per_second: int = field(default=60)
    rate_limit_concurrency: bool = field(default=True)
    rate_limit_per_minute: bool = field(default=False)
    rate_limit_per_second: bool = field(default=False)

    def count_tokens(self, text: str) -> int:
        """
        Returns the number of tokens for a given text using the encoding appropriate for the model.
        """
        return len(TOKEN_PATTERN.findall(text))


    def is_within_token_limit(self, text: str, token_limit: int):
        """
        Lightweight check to determine if `text` fits within `token_limit` tokens.
        Returns the token count (an int) if it is less than or equal to the limit,
        otherwise returns False.
        """
        token_count = self.count_tokens(text)
        return token_count if token_count <= token_limit else False

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
            model (str): The name of the model to use.
            system_prompt (str, optional): The system prompt to set the context for the conversation. Defaults to None.
            history_messages (list, optional): A list of previous messages in the conversation. Defaults to empty.
            response_model (Type[T], optional): The Pydantic model to parse the response. Defaults to None.
            **kwargs: Additional keyword arguments that may be required by specific LLM implementations.

        Returns:
            str: The response from the language model.
        """
        raise NotImplementedError


@dataclass
class BaseEmbeddingService:
    """Base class for Language Model implementations."""

    embedding_dim: int = field(default=1536)
    model: Optional[str] = field(default="text-embedding-3-small")
    base_url: Optional[str] = field(default=None)
    api_key: Optional[str] = field(default=None)
    max_requests_concurrent: int = field(default=int(os.getenv("CONCURRENT_TASK_LIMIT", 1024)))
    max_requests_per_minute: int = field(default=500) # Tier 1 OpenAI RPM
    max_requests_per_second: int = field(default=100)
    rate_limit_concurrency: bool = field(default=True)
    rate_limit_per_minute: bool = field(default=True)
    rate_limit_per_second: bool = field(default=False)

    embedding_async_client: Any = field(init=False, default=None)

    async def encode(
        self, texts: list[str], model: Optional[str] = None
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Get the embedding representation of the input text.

        Args:
            texts (str): The input text to embed.
            model (str): The name of the model to use.

        Returns:
            list[float]: The embedding vector as a list of floats.
        """
        raise NotImplementedError

class NoopAsyncContextManager:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass