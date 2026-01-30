"""Integration tests for the evaluation pipeline."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evalab.config.schemas import RunConfig
from evalab.data.loader import load_dataset
from evalab.generation.result import GenerationResult
from evalab.storage.database import init_db, reset_db


class TestDataLoading:
    """Integration tests for data loading."""

    def test_load_dataset_integration(self, sample_dataset_path):
        """Test loading a complete dataset."""
        dataset = load_dataset(sample_dataset_path)

        assert dataset.name == "test_dataset"
        assert len(dataset.examples) == 2
        assert dataset.hash is not None

        # Check example structure
        example = dataset.examples[0]
        assert example.id == "qa_001"
        assert example.task.value == "qa"

    def test_dataset_filtering(self, sample_dataset_path):
        """Test dataset filtering by task."""
        from evalab.data.schemas import TaskType

        dataset = load_dataset(sample_dataset_path)
        filtered = dataset.filter_by_task(TaskType.QA)

        assert len(filtered.examples) == len(dataset.examples)

    def test_dataset_max_examples(self, sample_dataset_path):
        """Test limiting number of examples."""
        dataset = load_dataset(sample_dataset_path, max_examples=1)

        assert len(dataset.examples) == 1


class TestDatabaseIntegration:
    """Integration tests for database operations."""

    def test_run_registry_lifecycle(self, temp_dir):
        """Test creating, updating, and querying runs."""
        from evalab.storage.registry import RunRegistry

        # Initialize fresh database
        db_path = temp_dir / "test.db"
        init_db(f"sqlite:///{db_path}")

        registry = RunRegistry(str(temp_dir / "runs"))

        # Create run
        run_id = registry.create_run(
            name="test_run",
            config={"model": "test"},
            notes="Test run",
            tags=["test"],
        )

        assert run_id is not None

        # Get run
        run = registry.get_run(run_id)
        assert run is not None
        assert run.name == "test_run"
        assert run.status == "pending"

        # Start run
        registry.start_run(run_id)
        run = registry.get_run(run_id)
        assert run.status == "running"

        # Add prediction
        registry.add_prediction(
            run_id=run_id,
            example_id="ex1",
            output_text="test output",
            latency_ms=100.0,
            input_tokens=10,
            output_tokens=5,
        )

        # Add metrics
        registry.add_metric(
            run_id=run_id,
            example_id="ex1",
            metric_name="accuracy",
            metric_value=1.0,
        )

        # Complete run
        registry.complete_run(run_id)
        run = registry.get_run(run_id)
        assert run.status == "completed"

        # List runs
        runs = registry.list_runs()
        assert len(runs) >= 1

        # Delete run
        deleted = registry.delete_run(run_id)
        assert deleted is True

        run = registry.get_run(run_id)
        assert run is None


class TestPipelineWithMockedLLM:
    """Integration tests for pipeline with mocked LLM."""

    @pytest.fixture
    def mock_llm_backend(self):
        """Create a mock LLM backend."""
        mock = MagicMock()
        mock.generate.return_value = GenerationResult(
            text="4",
            input_tokens=10,
            output_tokens=2,
            latency_ms=50.0,
            finish_reason="stop",
            model="mock-model",
        )
        mock.count_tokens.return_value = 10
        return mock

    def test_pipeline_with_mocked_llm(
        self,
        temp_dir,
        sample_dataset_path,
        mock_llm_backend,
    ):
        """Test running pipeline with mocked LLM."""
        from evalab.config.schemas import RunConfig
        from evalab.pipeline.runner import PipelineRunner

        # Create prompt template
        prompt_path = temp_dir / "prompt.jinja"
        prompt_path.write_text("Q: {{ question }}\nA:")

        # Create config
        config = RunConfig(
            run_name="mock_test",
            dataset={"path": str(sample_dataset_path)},
            generation={"backend": "openai", "model": "gpt-4o-mini"},
            prompt={"template_path": str(prompt_path)},
            evaluation={"suites": ["accuracy"]},
            logging={"save_traces": False, "output_dir": str(temp_dir / "runs")},
        )

        # Initialize database
        db_path = temp_dir / "test.db"
        init_db(f"sqlite:///{db_path}")

        # Create runner and mock the LLM
        runner = PipelineRunner(config)
        runner._llm = mock_llm_backend

        # Run pipeline
        result = runner.run()

        assert result.status == "completed"
        assert result.num_examples == 2
        assert "exact_match" in result.metrics_summary


class TestEvaluationSuites:
    """Integration tests for evaluation suites."""

    def test_accuracy_suite_integration(self):
        """Test accuracy suite with various inputs."""
        from evalab.evaluation.accuracy import AccuracySuite

        suite = AccuracySuite()

        # Test exact match
        results = suite.evaluate("Paris", {"answer": "Paris"})
        em = next(r for r in results if r.name == "exact_match")
        assert em.value == 1.0

        # Test partial match
        results = suite.evaluate("The capital is Paris", {"answer": "Paris"})
        f1 = next(r for r in results if r.name == "token_f1")
        assert f1.value > 0  # Should have some overlap

    def test_cost_latency_suite_integration(self):
        """Test cost/latency suite."""
        from evalab.evaluation.cost_latency import CostLatencySuite

        suite = CostLatencySuite(model_name="gpt-4o-mini")

        results = suite.evaluate(
            "test output",
            "reference",
            input_tokens=100,
            output_tokens=50,
            latency_ms=200.0,
        )

        latency = next(r for r in results if r.name == "latency_ms")
        assert latency.value == 200.0

        cost = next(r for r in results if r.name == "estimated_cost_usd")
        assert cost.value > 0
