"""vLLM backend for local GPU inference."""

import logging
import os
import time
from typing import Any

import httpx

from evalab.generation.base import LLMBackend
from evalab.generation.result import GenerationResult

logger = logging.getLogger(__name__)


class VLLMBackend(LLMBackend):
    """
    vLLM backend for local GPU inference.

    Connects to a running vLLM server via OpenAI-compatible API.
    """

    def __init__(
        self,
        model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        api_base: str | None = None,
        api_key: str = "EMPTY",
        timeout: float = 120.0,
    ):
        """
        Initialize vLLM backend.

        Args:
            model: Model name (must match vLLM server model)
            api_base: vLLM server URL (defaults to localhost:8000)
            api_key: API key (usually "EMPTY" for local vLLM)
            timeout: Request timeout in seconds
        """
        super().__init__(model)

        self._api_base = api_base or os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1")
        self._api_key = api_key
        self._timeout = timeout

        self._client = httpx.Client(
            base_url=self._api_base,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=timeout,
        )

        # Simple tokenizer approximation (assumes ~4 chars per token)
        # For accurate counting, use the model's tokenizer
        self._chars_per_token = 4

        logger.info(f"Initialized vLLM backend: {model} at {self._api_base}")

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
        Generate text using vLLM server.

        Args:
            prompt: User prompt
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            top_p: Top-p sampling
            seed: Random seed
            system_message: Optional system message
            **kwargs: Additional parameters

        Returns:
            GenerationResult with generated text
        """
        messages = []

        if system_message:
            messages.append({"role": "system", "content": system_message})

        messages.append({"role": "user", "content": prompt})

        start_time = time.time()

        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
            }

            if seed is not None:
                payload["seed"] = seed

            # Add any extra parameters
            payload.update(kwargs)

            response = self._client.post("/chat/completions", json=payload)
            response.raise_for_status()

            data = response.json()
            latency_ms = (time.time() - start_time) * 1000

            choice = data["choices"][0]
            usage = data.get("usage", {})

            return GenerationResult(
                text=choice["message"]["content"],
                input_tokens=usage.get("prompt_tokens", self.count_tokens(prompt)),
                output_tokens=usage.get("completion_tokens", 0),
                latency_ms=latency_ms,
                finish_reason=choice.get("finish_reason"),
                model=data.get("model", self.model),
                raw_response=data,
            )

        except httpx.HTTPStatusError as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"vLLM HTTP error: {e}")

            return GenerationResult(
                text="",
                input_tokens=self.count_tokens(prompt),
                output_tokens=0,
                latency_ms=latency_ms,
                error=f"HTTP {e.response.status_code}: {e.response.text}",
                model=self.model,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"vLLM generation error: {e}")

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
        Approximate token count.

        For accurate counting, use the model's tokenizer.

        Args:
            text: Input text

        Returns:
            Approximate token count
        """
        return len(text) // self._chars_per_token

    def health_check(self) -> bool:
        """
        Check if vLLM server is running.

        Returns:
            True if server is healthy
        """
        try:
            response = self._client.get("/models")
            return response.status_code == 200
        except Exception:
            return False

    def get_available_models(self) -> list[str]:
        """
        Get list of models available on the server.

        Returns:
            List of model names
        """
        try:
            response = self._client.get("/models")
            response.raise_for_status()
            data = response.json()
            return [m["id"] for m in data.get("data", [])]
        except Exception as e:
            logger.error(f"Failed to get models: {e}")
            return []


def create_backend(
    backend_type: str,
    model: str,
    **kwargs: Any,
) -> LLMBackend:
    """
    Factory function to create an LLM backend.

    Args:
        backend_type: One of "openai", "vllm"
        model: Model identifier
        **kwargs: Backend-specific arguments

    Returns:
        LLMBackend instance
    """
    from evalab.generation.openai_backend import OpenAIBackend

    if backend_type == "openai":
        return OpenAIBackend(model=model, **kwargs)
    elif backend_type == "vllm":
        return VLLMBackend(model=model, **kwargs)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
