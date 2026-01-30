"""Main pipeline runner for evaluation."""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evalab.attribution.taxonomy import ErrorClassifier
from evalab.config.schemas import EvaluationSuiteType, RunConfig
from evalab.data.loader import load_corpus, load_dataset
from evalab.data.schemas import Dataset, Example, TaskType
from evalab.evaluation.accuracy import AccuracySuite
from evalab.evaluation.base import MetricResult
from evalab.evaluation.calibration import CalibrationSuite
from evalab.evaluation.cost_latency import CostLatencySuite
from evalab.evaluation.faithfulness import FaithfulnessSuite
from evalab.evaluation.robustness import RobustnessSuite
from evalab.evaluation.semantic import SemanticSuite
from evalab.generation.base import LLMBackend
from evalab.generation.openai_backend import OpenAIBackend
from evalab.generation.prompt import PromptRenderer
from evalab.generation.result import GenerationResult, GenerationTrace
from evalab.generation.vllm_backend import VLLMBackend
from evalab.retrieval.base import BaseRetriever, RetrievalResult
from evalab.retrieval.chunker import Chunker
from evalab.retrieval.hybrid import HybridRetriever, create_retriever
from evalab.storage.database import init_db
from evalab.storage.registry import RunRegistry

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Result of a pipeline run."""

    run_id: str
    status: str
    num_examples: int
    metrics_summary: dict[str, Any]
    duration_sec: float
    error: str | None = None
    traces: list[GenerationTrace] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "status": self.status,
            "num_examples": self.num_examples,
            "metrics_summary": self.metrics_summary,
            "duration_sec": self.duration_sec,
            "error": self.error,
        }


class PipelineRunner:
    """
    Main evaluation pipeline runner.

    Orchestrates:
    1. Data loading
    2. Retrieval (if enabled)
    3. Generation
    4. Evaluation
    5. Attribution (if enabled)
    6. Storage
    """

    def __init__(
        self,
        config: RunConfig,
        registry: RunRegistry | None = None,
        db_url: str | None = None,
    ):
        """
        Initialize pipeline runner.

        Args:
            config: Run configuration
            registry: Optional run registry (created if not provided)
            db_url: Optional database URL
        """
        self.config = config
        self.registry = registry or RunRegistry(config.logging.output_dir)

        # Initialize database
        init_db(db_url)

        # Components (lazily initialized)
        self._llm: LLMBackend | None = None
        self._retriever: BaseRetriever | None = None
        self._prompt_renderer: PromptRenderer | None = None
        self._eval_suites: dict[str, Any] = {}

    def _init_llm(self) -> LLMBackend:
        """Initialize LLM backend."""
        if self._llm is None:
            gen_config = self.config.generation

            if gen_config.backend.value == "openai":
                self._llm = OpenAIBackend(
                    model=gen_config.model,
                    base_url=gen_config.api_base,
                )
            elif gen_config.backend.value == "vllm":
                self._llm = VLLMBackend(
                    model=gen_config.model,
                    api_base=gen_config.api_base,
                )
            else:
                raise ValueError(f"Unknown backend: {gen_config.backend}")

        return self._llm

    def _init_retriever(self, corpus_path: str) -> BaseRetriever:
        """Initialize retrieval system."""
        if self._retriever is None:
            ret_config = self.config.retrieval

            # Load corpus
            corpus = load_corpus(corpus_path)

            # Chunk documents
            chunker = Chunker(
                chunk_size=ret_config.chunking.size,
                overlap=ret_config.chunking.overlap,
            )

            chunks = chunker.chunk_documents(
                [{"doc_id": d.doc_id, "text": d.text, "title": d.title} for d in corpus.documents]
            )

            # Convert to dicts
            chunk_dicts = [c.to_dict() for c in chunks]

            # Create retriever
            self._retriever = create_retriever(
                retriever_type=ret_config.retriever.type.value,
                bm25_weight=ret_config.retriever.bm25_weight,
                dense_weight=ret_config.retriever.dense_weight,
                dense_model=ret_config.retriever.embedding_model,
            )

            # Index chunks
            self._retriever.index(chunk_dicts)
            logger.info(f"Indexed {len(chunk_dicts)} chunks")

        return self._retriever

    def _init_prompt_renderer(self) -> PromptRenderer:
        """Initialize prompt renderer."""
        if self._prompt_renderer is None:
            template_dir = Path(self.config.prompt.template_path).parent
            self._prompt_renderer = PromptRenderer(template_dir)

        return self._prompt_renderer

    def _init_eval_suites(self) -> dict[str, Any]:
        """Initialize evaluation suites."""
        if not self._eval_suites:
            suite_types = self.config.evaluation.suites

            if EvaluationSuiteType.ACCURACY in suite_types:
                self._eval_suites["accuracy"] = AccuracySuite()

            if EvaluationSuiteType.SEMANTIC in suite_types:
                self._eval_suites["semantic"] = SemanticSuite(
                    use_bertscore=True,
                    use_embeddings=True,
                )

            if EvaluationSuiteType.FAITHFULNESS in suite_types:
                self._eval_suites["faithfulness"] = FaithfulnessSuite(
                    similarity_threshold=self.config.evaluation.faithfulness_threshold,
                )

            if EvaluationSuiteType.ROBUSTNESS in suite_types:
                self._eval_suites["robustness"] = RobustnessSuite()

            if EvaluationSuiteType.CALIBRATION in suite_types:
                self._eval_suites["calibration"] = CalibrationSuite(
                    num_bins=self.config.evaluation.calibration_bins,
                )

            if EvaluationSuiteType.COST_LATENCY in suite_types:
                self._eval_suites["cost_latency"] = CostLatencySuite(
                    model_name=self.config.generation.model,
                )

        return self._eval_suites

    def _render_prompt(
        self,
        example: Example,
        context: str | None = None,
        chunks: list[dict[str, Any]] | None = None,
    ) -> str:
        """Render prompt for an example."""
        renderer = self._init_prompt_renderer()
        template_path = self.config.prompt.template_path

        # Get input data
        typed_input = example.get_typed_input()

        # Build template variables
        variables: dict[str, Any] = {}

        if example.task == TaskType.QA:
            variables["question"] = typed_input.question
            variables["context"] = context or typed_input.context or ""

        elif example.task == TaskType.RAG_QA:
            variables["question"] = typed_input.question
            variables["context"] = context or ""
            variables["chunks"] = chunks or []

        elif example.task == TaskType.SUMMARIZATION:
            variables["document"] = typed_input.document
            variables["max_length"] = typed_input.max_length

        elif example.task == TaskType.CLASSIFICATION:
            variables["text"] = typed_input.text
            variables["labels"] = typed_input.labels

        return renderer.render(template_path, **variables)

    def _generate(
        self,
        prompt: str,
        example_id: str,
    ) -> GenerationResult:
        """Generate output for a prompt."""
        llm = self._init_llm()
        gen_config = self.config.generation

        result = llm.generate(
            prompt=prompt,
            temperature=gen_config.temperature,
            max_tokens=gen_config.max_tokens,
            top_p=gen_config.top_p,
            seed=gen_config.seed,
            system_message=self.config.prompt.system_message,
        )

        return result

    def _evaluate_example(
        self,
        example: Example,
        prediction: str,
        generation_result: GenerationResult,
        retrieval_result: RetrievalResult | None = None,
    ) -> list[MetricResult]:
        """Evaluate a single example."""
        suites = self._init_eval_suites()
        all_metrics: list[MetricResult] = []

        # Get reference
        reference = example.reference

        # Prepare context for evaluation
        context = retrieval_result.get_context() if retrieval_result else None
        chunks = [c.to_dict() for c in retrieval_result.chunks] if retrieval_result else []

        for suite_name, suite in suites.items():
            try:
                if suite_name == "cost_latency":
                    metrics = suite.evaluate(
                        prediction,
                        reference,
                        input_tokens=generation_result.input_tokens,
                        output_tokens=generation_result.output_tokens,
                        latency_ms=generation_result.latency_ms,
                        model=generation_result.model,
                    )
                elif suite_name == "faithfulness":
                    metrics = suite.evaluate(
                        prediction,
                        reference,
                        context=context,
                        chunks=chunks,
                    )
                else:
                    metrics = suite.evaluate(prediction, reference)

                all_metrics.extend(metrics)

            except Exception as e:
                logger.warning(f"Evaluation suite {suite_name} failed: {e}")

        return all_metrics

    def run(self) -> RunResult:
        """
        Run the evaluation pipeline.

        Returns:
            RunResult with metrics and status
        """
        start_time = time.time()

        # Create run
        run_id = self.registry.create_run(
            name=self.config.run_name,
            config=self.config.model_dump(),
            notes=self.config.notes,
            tags=self.config.tags,
        )

        logger.info(f"Starting run: {run_id}")
        self.registry.start_run(run_id)

        try:
            # Load dataset
            dataset = load_dataset(
                self.config.dataset.path,
                name=self.config.dataset.name,
                max_examples=self.config.dataset.max_examples,
            )

            logger.info(f"Loaded {len(dataset.examples)} examples")

            # Initialize retriever if needed
            retriever = None
            if self.config.retrieval.enabled and self.config.retrieval.corpus_path:
                retriever = self._init_retriever(self.config.retrieval.corpus_path)

            # Process examples
            all_metrics: list[dict[str, Any]] = []
            all_latencies: list[float] = []
            all_input_tokens: list[int] = []
            all_output_tokens: list[int] = []
            traces: list[GenerationTrace] = []

            for i, example in enumerate(dataset.examples):
                logger.debug(f"Processing example {i + 1}/{len(dataset.examples)}: {example.id}")

                # Retrieval
                retrieval_result = None
                context = None
                chunks = None

                if retriever and example.task in (TaskType.RAG_QA, TaskType.QA):
                    typed_input = example.get_typed_input()
                    query = typed_input.question
                    retrieval_result = retriever.retrieve(query, top_k=self.config.retrieval.top_k)
                    context = retrieval_result.get_context()
                    chunks = [c.to_dict() for c in retrieval_result.chunks]

                    # Save retrieval trace
                    if self.config.logging.save_traces:
                        self.registry.artifacts.append_retrieval_trace(
                            run_id, retrieval_result.to_dict()
                        )

                # Render prompt
                prompt = self._render_prompt(example, context=context, chunks=chunks)

                # Generate
                generation_result = self._generate(prompt, example.id)

                # Save prediction
                self.registry.add_prediction(
                    run_id=run_id,
                    example_id=example.id,
                    output_text=generation_result.text,
                    latency_ms=generation_result.latency_ms,
                    input_tokens=generation_result.input_tokens,
                    output_tokens=generation_result.output_tokens,
                    finish_reason=generation_result.finish_reason,
                    error=generation_result.error,
                )

                # Track for aggregation
                all_latencies.append(generation_result.latency_ms)
                all_input_tokens.append(generation_result.input_tokens)
                all_output_tokens.append(generation_result.output_tokens)

                # Evaluate
                metrics = self._evaluate_example(
                    example,
                    generation_result.text,
                    generation_result,
                    retrieval_result,
                )

                # Save metrics
                for metric in metrics:
                    self.registry.add_metric(
                        run_id=run_id,
                        example_id=example.id,
                        metric_name=metric.name,
                        metric_value=metric.value,
                        metric_json=metric.details if metric.details else None,
                    )
                    all_metrics.append({
                        "example_id": example.id,
                        "metric_name": metric.name,
                        "metric_value": metric.value,
                    })

                # Save trace
                if self.config.logging.save_traces:
                    trace = GenerationTrace(
                        example_id=example.id,
                        prompt=prompt,
                        result=generation_result,
                        model=self.config.generation.model,
                        temperature=self.config.generation.temperature,
                        max_tokens=self.config.generation.max_tokens,
                        seed=self.config.generation.seed,
                        system_message=self.config.prompt.system_message,
                        retrieved_context=context,
                    )
                    traces.append(trace)

            # Compute aggregates
            self.registry.compute_aggregates(run_id)

            # Compute dataset-level cost/latency metrics
            cost_suite = self._eval_suites.get("cost_latency")
            if cost_suite:
                duration_sec = time.time() - start_time
                dataset_metrics = cost_suite.evaluate_dataset(
                    all_input_tokens,
                    all_output_tokens,
                    all_latencies,
                    model=self.config.generation.model,
                    total_duration_sec=duration_sec,
                )
                for metric in dataset_metrics:
                    self.registry.add_metric(
                        run_id=run_id,
                        example_id="__aggregate__",
                        metric_name=metric.name,
                        metric_value=metric.value,
                        metric_json=metric.details if metric.details else None,
                    )

            # Complete run
            self.registry.complete_run(run_id)

            duration_sec = time.time() - start_time
            logger.info(f"Run {run_id} completed in {duration_sec:.2f}s")

            # Build summary
            metrics_summary = self._build_metrics_summary(all_metrics)

            return RunResult(
                run_id=run_id,
                status="completed",
                num_examples=len(dataset.examples),
                metrics_summary=metrics_summary,
                duration_sec=duration_sec,
                traces=traces if self.config.logging.save_traces else [],
            )

        except Exception as e:
            logger.error(f"Run {run_id} failed: {e}")
            self.registry.complete_run(run_id, error=str(e))

            return RunResult(
                run_id=run_id,
                status="failed",
                num_examples=0,
                metrics_summary={},
                duration_sec=time.time() - start_time,
                error=str(e),
            )

    def _build_metrics_summary(
        self,
        all_metrics: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build metrics summary from all example metrics."""
        import numpy as np

        summary: dict[str, Any] = {}

        # Group by metric name
        by_name: dict[str, list[float]] = {}
        for m in all_metrics:
            name = m["metric_name"]
            if name not in by_name:
                by_name[name] = []
            by_name[name].append(m["metric_value"])

        # Compute stats
        for name, values in by_name.items():
            arr = np.array(values)
            summary[name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "count": len(values),
            }

        return summary
