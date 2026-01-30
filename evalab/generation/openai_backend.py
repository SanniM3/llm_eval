"""OpenAI API backend."""

import logging
import os
import time
from typing import Any

import tiktoken
from openai import OpenAI

from evalab.generation.base import LLMBackend
from evalab.generation.result import GenerationResult

logger = logging.getLogger(__name__)


class OpenAIBackend(LLMBackend):
    """
    OpenAI API backend for generation.

    Supports GPT-4, GPT-4o, GPT-3.5-turbo, and other OpenAI models.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        organization: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        """
        Initialize OpenAI backend.

        Args:
            model: Model name (e.g., "gpt-4o-mini", "gpt-4o")
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            organization: Optional organization ID
            base_url: Optional custom API base URL
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
        """
        super().__init__(model)

        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass api_key.")

        self._client = OpenAI(
            api_key=self._api_key,
            organization=organization,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

        # Initialize tokenizer
        try:
            self._tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback for unknown models
            self._tokenizer = tiktoken.get_encoding("cl100k_base")

        logger.info(f"Initialized OpenAI backend with model: {model}")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
        top_p: float = 1.0,
        seed: int | None = None,
        system_message: str | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """
        Generate text using OpenAI API.

        Args:
            prompt: User prompt
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum output tokens
            top_p: Top-p sampling
            seed: Random seed for reproducibility
            system_message: Optional system message
            **kwargs: Additional parameters (e.g., stop sequences)

        Returns:
            GenerationResult with generated text
        """
        messages = []

        if system_message:
            messages.append({"role": "system", "content": system_message})

        messages.append({"role": "user", "content": prompt})

        start_time = time.time()

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                seed=seed,
                **kwargs,
            )

            latency_ms = (time.time() - start_time) * 1000

            choice = response.choices[0]
            usage = response.usage

            return GenerationResult(
                text=choice.message.content or "",
                input_tokens=usage.prompt_tokens if usage else self.count_tokens(prompt),
                output_tokens=usage.completion_tokens if usage else 0,
                latency_ms=latency_ms,
                finish_reason=choice.finish_reason,
                model=response.model,
                raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"OpenAI generation error: {e}")

            return GenerationResult(
                text="",
                input_tokens=self.count_tokens(prompt),
                output_tokens=0,
                latency_ms=latency_ms,
                error=str(e),
                model=self.model,
            )

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken.

        Args:
            text: Input text

        Returns:
            Token count
        """
        return len(self._tokenizer.encode(text))

    def generate_with_json(
        self,
        prompt: str,
        json_schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """
        Generate JSON-formatted output.

        Args:
            prompt: User prompt
            json_schema: Optional JSON schema for structured output
            **kwargs: Additional generation parameters

        Returns:
            GenerationResult with JSON text
        """
        response_format = {"type": "json_object"}

        return self.generate(
            prompt=prompt,
            response_format=response_format,
            **kwargs,
        )
