"""Base LLM backend interface."""

from abc import ABC, abstractmethod
from typing import Any

from evalab.generation.result import GenerationResult


class LLMBackend(ABC):
    """
    Abstract base class for LLM backends.

    Implementations should handle connection, generation, and token counting.
    """

    def __init__(self, model: str, **kwargs: Any):
        """
        Initialize LLM backend.

        Args:
            model: Model identifier
            **kwargs: Additional configuration
        """
        self.model = model
        self._config = kwargs

    @abstractmethod
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
        Generate text from a prompt.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            top_p: Top-p sampling parameter
            seed: Random seed for reproducibility
            system_message: Optional system message
            **kwargs: Additional model-specific parameters

        Returns:
            GenerationResult with generated text and metadata
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Input text

        Returns:
            Token count
        """
        pass

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Dictionary with model metadata
        """
        return {
            "model": self.model,
            "backend": self.__class__.__name__,
            "config": self._config,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"
