"""Robustness evaluation suite with perturbations."""

import logging
import random
import re
from typing import Any, Callable

import numpy as np

from evalab.evaluation.base import EvaluationSuite, MetricResult

logger = logging.getLogger(__name__)


class Perturbation:
    """Base class for input perturbations."""

    def __init__(self, name: str):
        self.name = name

    def apply(self, text: str, **kwargs: Any) -> str:
        """Apply perturbation to text."""
        raise NotImplementedError


class ParaphrasePerturbation(Perturbation):
    """Paraphrase input using LLM."""

    def __init__(self, llm_backend: Any | None = None):
        super().__init__(name="paraphrase")
        self.llm = llm_backend

    def apply(self, text: str, **kwargs: Any) -> str:
        if self.llm is None:
            # Simple word-level perturbation as fallback
            return self._simple_paraphrase(text)

        prompt = f"Paraphrase the following text while keeping the exact same meaning:\n\n{text}\n\nParaphrased:"
        result = self.llm.generate(prompt, temperature=0.7, max_tokens=256)

        if result.is_error:
            return self._simple_paraphrase(text)

        return result.text.strip()

    def _simple_paraphrase(self, text: str) -> str:
        """Simple word substitution as fallback."""
        substitutions = {
            "what": "which",
            "how": "in what way",
            "why": "for what reason",
            "when": "at what time",
            "where": "in what location",
            "is": "represents",
            "are": "represent",
        }

        words = text.lower().split()
        for i, word in enumerate(words):
            clean_word = re.sub(r"[^\w]", "", word)
            if clean_word in substitutions and random.random() > 0.5:
                words[i] = word.replace(clean_word, substitutions[clean_word])

        return " ".join(words)


class TypoPerturbation(Perturbation):
    """Add typos to input."""

    def __init__(self, error_rate: float = 0.1):
        super().__init__(name="typos")
        self.error_rate = error_rate

    def apply(self, text: str, **kwargs: Any) -> str:
        chars = list(text)
        keyboard_neighbors = {
            "a": "sqwz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdr",
            "f": "drtgvc", "g": "ftyhbv", "h": "gyujbn", "i": "ujko", "j": "huiknm",
            "k": "jiolm", "l": "kop", "m": "njk", "n": "bhjm", "o": "iklp",
            "p": "ol", "q": "wa", "r": "edft", "s": "wazxde", "t": "rfgy",
            "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu",
            "z": "asx",
        }

        for i, char in enumerate(chars):
            if random.random() < self.error_rate and char.lower() in keyboard_neighbors:
                neighbors = keyboard_neighbors[char.lower()]
                replacement = random.choice(neighbors)
                if char.isupper():
                    replacement = replacement.upper()
                chars[i] = replacement

        return "".join(chars)


class DistractorPerturbation(Perturbation):
    """Add distractor sentence to context."""

    def __init__(self, distractors: list[str] | None = None):
        super().__init__(name="distractor")
        self.distractors = distractors or [
            "The weather today is partly cloudy with a chance of rain.",
            "Many people enjoy outdoor activities during the summer months.",
            "Technology has advanced significantly in recent decades.",
            "Coffee is one of the most popular beverages worldwide.",
            "The global economy has experienced various fluctuations over time.",
        ]

    def apply(self, text: str, **kwargs: Any) -> str:
        distractor = random.choice(self.distractors)
        sentences = text.split(". ")

        if len(sentences) > 1:
            insert_pos = random.randint(1, len(sentences) - 1)
            sentences.insert(insert_pos, distractor)
            return ". ".join(sentences)
        else:
            return f"{text} {distractor}"


class DropTopKPerturbation(Perturbation):
    """Drop the top-k retrieved document."""

    def __init__(self, k: int = 1):
        super().__init__(name="drop_top_k")
        self.k = k

    def apply(self, text: str, **kwargs: Any) -> str:
        chunks = kwargs.get("chunks", [])
        if not chunks or len(chunks) <= self.k:
            return text

        # Remove top k chunks
        remaining = chunks[self.k:]
        return "\n\n".join(
            c.get("text", "") if isinstance(c, dict) else str(c) for c in remaining
        )


class RobustnessSuite(EvaluationSuite):
    """
    Robustness evaluation suite.

    Measures stability under input perturbations:
    - Output consistency (semantic similarity)
    - Decision flip rate (for classification)
    - Faithfulness delta
    """

    def __init__(
        self,
        perturbations: list[str] | None = None,
        llm_backend: Any | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize robustness suite.

        Args:
            perturbations: List of perturbation types to apply
            llm_backend: LLM backend for paraphrase perturbation
            embedding_model: Model for similarity computation
        """
        super().__init__(name="robustness")
        self.embedding_model = embedding_model

        # Initialize perturbations
        self._perturbations: dict[str, Perturbation] = {
            "paraphrase": ParaphrasePerturbation(llm_backend),
            "typos": TypoPerturbation(),
            "distractor": DistractorPerturbation(),
            "drop_top_k": DropTopKPerturbation(),
        }

        self.active_perturbations = perturbations or ["paraphrase", "typos", "distractor"]

        self._embedder = None

    def _get_embedder(self) -> Any:
        """Lazy load sentence transformer."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.embedding_model)
        return self._embedder

    def _compute_consistency(self, original: str, perturbed: str) -> float:
        """
        Compute output consistency as semantic similarity.

        Args:
            original: Original output
            perturbed: Output after perturbation

        Returns:
            Similarity score (0-1)
        """
        embedder = self._get_embedder()

        embeddings = embedder.encode([original, perturbed], normalize_embeddings=True)
        similarity = float(np.dot(embeddings[0], embeddings[1]))

        return max(0.0, similarity)

    def _check_decision_flip(
        self,
        original: str,
        perturbed: str,
        normalize: bool = True,
    ) -> bool:
        """
        Check if decision changed after perturbation.

        Args:
            original: Original output
            perturbed: Perturbed output
            normalize: Whether to normalize outputs

        Returns:
            True if decision flipped
        """
        if normalize:
            original = original.strip().lower()
            perturbed = perturbed.strip().lower()

        return original != perturbed

    def evaluate(
        self,
        prediction: str,
        reference: str | dict[str, Any],
        **kwargs: Any,
    ) -> list[MetricResult]:
        """
        Evaluate robustness.

        Note: This method computes metrics based on original and perturbed outputs.
        The actual perturbation and re-generation should be done by the pipeline.

        Args:
            prediction: Original model output
            reference: Reference (may include perturbed outputs)
            **kwargs:
                - perturbed_outputs: dict of perturbation_name -> output
                - task_type: Task type for decision flip computation

        Returns:
            List of robustness metrics
        """
        perturbed_outputs = kwargs.get("perturbed_outputs", {})
        task_type = kwargs.get("task_type", "qa")

        if not perturbed_outputs:
            return [
                MetricResult(
                    name="output_consistency",
                    value=1.0,
                    details={"note": "no perturbations provided"},
                )
            ]

        results = []
        consistencies = []
        flip_count = 0

        for pert_name, perturbed_output in perturbed_outputs.items():
            # Compute consistency
            consistency = self._compute_consistency(prediction, perturbed_output)
            consistencies.append(consistency)

            # Check decision flip
            if task_type == "classification":
                if self._check_decision_flip(prediction, perturbed_output):
                    flip_count += 1

            results.append(
                MetricResult(
                    name=f"consistency_{pert_name}",
                    value=consistency,
                    details={"perturbation": pert_name},
                )
            )

        # Aggregate metrics
        avg_consistency = float(np.mean(consistencies)) if consistencies else 1.0
        results.append(
            MetricResult(
                name="output_consistency",
                value=avg_consistency,
                details={"per_perturbation": dict(zip(perturbed_outputs.keys(), consistencies))},
            )
        )

        if task_type == "classification" and perturbed_outputs:
            flip_rate = flip_count / len(perturbed_outputs)
            results.append(
                MetricResult(
                    name="decision_flip_rate",
                    value=flip_rate,
                    details={"flips": flip_count, "total": len(perturbed_outputs)},
                )
            )

        return results

    def generate_perturbations(
        self,
        text: str,
        perturbation_types: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, str]:
        """
        Generate perturbed versions of input text.

        Args:
            text: Original input text
            perturbation_types: Types of perturbations to apply
            **kwargs: Additional context for perturbations

        Returns:
            Dict of perturbation_name -> perturbed_text
        """
        types = perturbation_types or self.active_perturbations
        perturbed = {}

        for pert_type in types:
            if pert_type in self._perturbations:
                try:
                    perturbed[pert_type] = self._perturbations[pert_type].apply(text, **kwargs)
                except Exception as e:
                    logger.warning(f"Perturbation {pert_type} failed: {e}")

        return perturbed

    def get_metric_names(self) -> list[str]:
        return ["output_consistency", "decision_flip_rate"]
