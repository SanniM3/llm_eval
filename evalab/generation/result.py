"""Generation result data structures."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationResult:
    """
    Result of an LLM generation call.

    Contains the generated text along with metadata about the generation.
    """

    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    finish_reason: str | None = None
    model: str | None = None
    raw_response: dict[str, Any] | None = None
    error: str | None = None

    @property
    def total_tokens(self) -> int:
        """Total tokens used (input + output)."""
        return self.input_tokens + self.output_tokens

    @property
    def is_error(self) -> bool:
        """Check if generation resulted in an error."""
        return self.error is not None

    @property
    def is_complete(self) -> bool:
        """Check if generation completed normally."""
        return self.finish_reason in ("stop", "end_turn", None) and not self.is_error

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
            "finish_reason": self.finish_reason,
            "model": self.model,
            "error": self.error,
        }


@dataclass
class GenerationTrace:
    """
    Detailed trace of a generation for debugging and analysis.

    Includes the full prompt, response, and all parameters.
    """

    example_id: str
    prompt: str
    result: GenerationResult
    model: str
    temperature: float
    max_tokens: int
    seed: int | None = None
    system_message: str | None = None
    retrieved_context: str | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "example_id": self.example_id,
            "prompt": self.prompt,
            "result": self.result.to_dict(),
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "system_message": self.system_message,
            "retrieved_context": self.retrieved_context,
            "extra_params": self.extra_params,
        }
