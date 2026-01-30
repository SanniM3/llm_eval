"""Counterfactual probes for error attribution."""

import logging
from dataclasses import dataclass, field
from typing import Any

from evalab.attribution.taxonomy import ErrorClassifier, ErrorType

logger = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    """Result of a counterfactual probe."""

    probe_name: str
    example_id: str
    original_correct: bool
    counterfactual_correct: bool
    accuracy_delta: float  # positive = improvement
    faithfulness_delta: float | None = None
    latency_delta: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "probe_name": self.probe_name,
            "example_id": self.example_id,
            "original_correct": self.original_correct,
            "counterfactual_correct": self.counterfactual_correct,
            "accuracy_delta": self.accuracy_delta,
            "faithfulness_delta": self.faithfulness_delta,
            "latency_delta": self.latency_delta,
            "details": self.details,
        }


class CounterfactualProbe:
    """
    Base class for counterfactual probes.

    Probes run controlled experiments to isolate error sources.
    """

    def __init__(self, name: str, llm_backend: Any = None):
        """
        Initialize probe.

        Args:
            name: Probe identifier
            llm_backend: LLM backend for generation
        """
        self.name = name
        self.llm_backend = llm_backend

    def run(
        self,
        example: dict[str, Any],
        original_result: dict[str, Any],
        **kwargs: Any,
    ) -> ProbeResult:
        """
        Run the probe on an example.

        Args:
            example: Original example data
            original_result: Original generation result
            **kwargs: Additional context

        Returns:
            ProbeResult with comparison
        """
        raise NotImplementedError


class GoldContextProbe(CounterfactualProbe):
    """
    Gold context probe.

    Fixes retrieval by using gold/reference context,
    then reruns generation to see if it improves.
    """

    def __init__(self, llm_backend: Any, prompt_renderer: Any = None):
        """
        Initialize gold context probe.

        Args:
            llm_backend: LLM backend
            prompt_renderer: Prompt template renderer
        """
        super().__init__(name="gold_context", llm_backend=llm_backend)
        self.prompt_renderer = prompt_renderer

    def run(
        self,
        example: dict[str, Any],
        original_result: dict[str, Any],
        **kwargs: Any,
    ) -> ProbeResult:
        """
        Run gold context probe.

        Uses reference supporting facts as context instead of retrieval.

        Args:
            example: Example with reference.supporting_facts
            original_result: Original result with is_correct, faithfulness_score

        Returns:
            ProbeResult comparing original vs gold context
        """
        example_id = example.get("id", "unknown")
        original_correct = original_result.get("is_correct", False)

        # Get gold context
        reference = example.get("reference", {})
        supporting_facts = reference.get("supporting_facts", [])

        if not supporting_facts:
            # No gold context available
            return ProbeResult(
                probe_name=self.name,
                example_id=example_id,
                original_correct=original_correct,
                counterfactual_correct=original_correct,
                accuracy_delta=0.0,
                details={"error": "no supporting facts available"},
            )

        # Build gold context
        gold_context = "\n\n".join(
            f"[{fact.get('doc_id', i)}] {fact.get('span', '')}"
            for i, fact in enumerate(supporting_facts, 1)
        )

        # Get question
        input_data = example.get("input", {})
        question = input_data.get("question", "")

        # Render prompt with gold context
        if self.prompt_renderer:
            prompt = self.prompt_renderer.render_string(
                """Answer the question based on the provided context.

Context:
{{ context }}

Question: {{ question }}

Answer:""",
                context=gold_context,
                question=question,
            )
        else:
            prompt = f"Context:\n{gold_context}\n\nQuestion: {question}\n\nAnswer:"

        # Generate with gold context
        result = self.llm_backend.generate(prompt, temperature=0.2)

        if result.is_error:
            return ProbeResult(
                probe_name=self.name,
                example_id=example_id,
                original_correct=original_correct,
                counterfactual_correct=original_correct,
                accuracy_delta=0.0,
                details={"error": result.error},
            )

        # Check correctness of counterfactual output
        # (simplified - in practice would use same evaluation)
        ref_answer = reference.get("answer", "")
        cf_output = result.text.strip().lower()
        ref_lower = ref_answer.lower()

        counterfactual_correct = ref_lower in cf_output or cf_output in ref_lower

        accuracy_delta = int(counterfactual_correct) - int(original_correct)

        return ProbeResult(
            probe_name=self.name,
            example_id=example_id,
            original_correct=original_correct,
            counterfactual_correct=counterfactual_correct,
            accuracy_delta=float(accuracy_delta),
            latency_delta=result.latency_ms - original_result.get("latency_ms", 0),
            details={
                "gold_context": gold_context[:500],
                "counterfactual_output": result.text[:500],
            },
        )


class VaryRetrieverProbe(CounterfactualProbe):
    """
    Vary retriever probe.

    Fixes prompt and model, varies retriever type to isolate
    retrieval contribution.
    """

    def __init__(
        self,
        llm_backend: Any,
        retrievers: dict[str, Any],
        prompt_renderer: Any = None,
    ):
        """
        Initialize vary retriever probe.

        Args:
            llm_backend: LLM backend
            retrievers: Dict of retriever_name -> retriever
            prompt_renderer: Prompt template renderer
        """
        super().__init__(name="vary_retriever", llm_backend=llm_backend)
        self.retrievers = retrievers
        self.prompt_renderer = prompt_renderer

    def run(
        self,
        example: dict[str, Any],
        original_result: dict[str, Any],
        **kwargs: Any,
    ) -> ProbeResult:
        """
        Run vary retriever probe.

        Args:
            example: Original example
            original_result: Result with original retriever
            **kwargs: Additional context

        Returns:
            ProbeResult with per-retriever comparison
        """
        example_id = example.get("id", "unknown")
        original_correct = original_result.get("is_correct", False)
        original_retriever = original_result.get("retriever_type", "unknown")

        results_by_retriever = {}
        question = example.get("input", {}).get("question", "")
        reference = example.get("reference", {})
        ref_answer = reference.get("answer", "")

        for retriever_name, retriever in self.retrievers.items():
            if retriever_name == original_retriever:
                results_by_retriever[retriever_name] = {
                    "correct": original_correct,
                    "is_original": True,
                }
                continue

            try:
                # Retrieve with different retriever
                retrieval_result = retriever.retrieve(question, top_k=5)
                context = retrieval_result.get_context()

                # Generate
                if self.prompt_renderer:
                    prompt = self.prompt_renderer.render_string(
                        """Answer based on the context.

Context: {{ context }}

Question: {{ question }}

Answer:""",
                        context=context,
                        question=question,
                    )
                else:
                    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

                result = self.llm_backend.generate(prompt, temperature=0.2)

                if not result.is_error:
                    output = result.text.strip().lower()
                    ref_lower = ref_answer.lower()
                    is_correct = ref_lower in output or output in ref_lower

                    results_by_retriever[retriever_name] = {
                        "correct": is_correct,
                        "is_original": False,
                        "latency_ms": result.latency_ms,
                    }
                else:
                    results_by_retriever[retriever_name] = {
                        "correct": False,
                        "error": result.error,
                    }

            except Exception as e:
                results_by_retriever[retriever_name] = {
                    "correct": False,
                    "error": str(e),
                }

        # Find best retriever
        best_retriever = max(
            results_by_retriever.keys(),
            key=lambda k: int(results_by_retriever[k].get("correct", False)),
        )
        best_correct = results_by_retriever[best_retriever].get("correct", False)

        accuracy_delta = int(best_correct) - int(original_correct)

        return ProbeResult(
            probe_name=self.name,
            example_id=example_id,
            original_correct=original_correct,
            counterfactual_correct=best_correct,
            accuracy_delta=float(accuracy_delta),
            details={
                "original_retriever": original_retriever,
                "best_retriever": best_retriever,
                "results_by_retriever": results_by_retriever,
            },
        )


class VaryPromptProbe(CounterfactualProbe):
    """
    Vary prompt probe.

    Fixes retrieval and model, varies prompt template to isolate
    prompt contribution.
    """

    def __init__(
        self,
        llm_backend: Any,
        prompt_templates: dict[str, str],
        prompt_renderer: Any = None,
    ):
        """
        Initialize vary prompt probe.

        Args:
            llm_backend: LLM backend
            prompt_templates: Dict of template_name -> template_string
            prompt_renderer: Prompt template renderer
        """
        super().__init__(name="vary_prompt", llm_backend=llm_backend)
        self.prompt_templates = prompt_templates
        self.prompt_renderer = prompt_renderer

    def run(
        self,
        example: dict[str, Any],
        original_result: dict[str, Any],
        **kwargs: Any,
    ) -> ProbeResult:
        """
        Run vary prompt probe.

        Args:
            example: Original example
            original_result: Result with original prompt
            **kwargs: Must include 'context' if RAG

        Returns:
            ProbeResult with per-prompt comparison
        """
        example_id = example.get("id", "unknown")
        original_correct = original_result.get("is_correct", False)
        original_prompt = original_result.get("prompt_template", "unknown")

        results_by_prompt = {}
        context = kwargs.get("context", "")
        question = example.get("input", {}).get("question", "")
        reference = example.get("reference", {})
        ref_answer = reference.get("answer", "")

        from evalab.generation.prompt import PromptRenderer

        renderer = self.prompt_renderer or PromptRenderer()

        for template_name, template in self.prompt_templates.items():
            if template_name == original_prompt:
                results_by_prompt[template_name] = {
                    "correct": original_correct,
                    "is_original": True,
                }
                continue

            try:
                # Render prompt with different template
                prompt = renderer.render_string(
                    template,
                    context=context,
                    question=question,
                    chunks=[{"text": context}] if context else [],
                )

                result = self.llm_backend.generate(prompt, temperature=0.2)

                if not result.is_error:
                    output = result.text.strip().lower()
                    ref_lower = ref_answer.lower()
                    is_correct = ref_lower in output or output in ref_lower

                    results_by_prompt[template_name] = {
                        "correct": is_correct,
                        "is_original": False,
                        "latency_ms": result.latency_ms,
                    }
                else:
                    results_by_prompt[template_name] = {
                        "correct": False,
                        "error": result.error,
                    }

            except Exception as e:
                results_by_prompt[template_name] = {
                    "correct": False,
                    "error": str(e),
                }

        # Find best prompt
        best_prompt = max(
            results_by_prompt.keys(),
            key=lambda k: int(results_by_prompt[k].get("correct", False)),
        )
        best_correct = results_by_prompt[best_prompt].get("correct", False)

        accuracy_delta = int(best_correct) - int(original_correct)

        return ProbeResult(
            probe_name=self.name,
            example_id=example_id,
            original_correct=original_correct,
            counterfactual_correct=best_correct,
            accuracy_delta=float(accuracy_delta),
            details={
                "original_prompt": original_prompt,
                "best_prompt": best_prompt,
                "results_by_prompt": results_by_prompt,
            },
        )


def summarize_probe_results(results: list[ProbeResult]) -> dict[str, Any]:
    """
    Summarize results from counterfactual probes.

    Args:
        results: List of ProbeResult objects

    Returns:
        Summary statistics
    """
    if not results:
        return {"total": 0}

    total = len(results)

    # Count improvements
    improvements = sum(1 for r in results if r.accuracy_delta > 0)
    regressions = sum(1 for r in results if r.accuracy_delta < 0)
    unchanged = total - improvements - regressions

    # Average deltas
    avg_accuracy_delta = sum(r.accuracy_delta for r in results) / total

    return {
        "total": total,
        "improvements": improvements,
        "regressions": regressions,
        "unchanged": unchanged,
        "improvement_rate": improvements / total,
        "avg_accuracy_delta": avg_accuracy_delta,
    }
