"""Artifact storage for run outputs and traces."""

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class ArtifactStore:
    """
    Manages file artifacts for evaluation runs.

    Directory structure:
        runs/<run_id>/
            config.yaml
            predictions.jsonl
            retrieval_traces.jsonl
            metrics.jsonl
            reports/
                attribution.md
                summary.md
    """

    def __init__(self, base_dir: str | Path = "runs"):
        """
        Initialize artifact store.

        Args:
            base_dir: Base directory for all run artifacts
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_run_dir(self, run_id: str) -> Path:
        """Get the directory for a specific run."""
        return self.base_dir / run_id

    def create_run_dir(self, run_id: str) -> Path:
        """
        Create directory structure for a new run.

        Args:
            run_id: Unique run identifier

        Returns:
            Path to the run directory
        """
        run_dir = self.get_run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "reports").mkdir(exist_ok=True)
        return run_dir

    def save_config(self, run_id: str, config: dict[str, Any]) -> Path:
        """
        Save run configuration to YAML file.

        Args:
            run_id: Run identifier
            config: Configuration dictionary

        Returns:
            Path to saved config file
        """
        run_dir = self.create_run_dir(run_id)
        config_path = run_dir / "config.yaml"

        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.debug(f"Saved config to {config_path}")
        return config_path

    def load_config(self, run_id: str) -> dict[str, Any]:
        """
        Load run configuration from YAML file.

        Args:
            run_id: Run identifier

        Returns:
            Configuration dictionary
        """
        config_path = self.get_run_dir(run_id) / "config.yaml"

        with open(config_path) as f:
            return yaml.safe_load(f)

    def append_prediction(self, run_id: str, prediction: dict[str, Any]) -> None:
        """
        Append a prediction to the predictions JSONL file.

        Args:
            run_id: Run identifier
            prediction: Prediction data to append
        """
        run_dir = self.create_run_dir(run_id)
        predictions_path = run_dir / "predictions.jsonl"

        with open(predictions_path, "a") as f:
            f.write(json.dumps(prediction) + "\n")

    def save_predictions(self, run_id: str, predictions: list[dict[str, Any]]) -> Path:
        """
        Save all predictions to JSONL file.

        Args:
            run_id: Run identifier
            predictions: List of prediction dictionaries

        Returns:
            Path to saved predictions file
        """
        run_dir = self.create_run_dir(run_id)
        predictions_path = run_dir / "predictions.jsonl"

        with open(predictions_path, "w") as f:
            for pred in predictions:
                f.write(json.dumps(pred) + "\n")

        logger.debug(f"Saved {len(predictions)} predictions to {predictions_path}")
        return predictions_path

    def load_predictions(self, run_id: str) -> list[dict[str, Any]]:
        """
        Load predictions from JSONL file.

        Args:
            run_id: Run identifier

        Returns:
            List of prediction dictionaries
        """
        predictions_path = self.get_run_dir(run_id) / "predictions.jsonl"
        predictions = []

        if predictions_path.exists():
            with open(predictions_path) as f:
                for line in f:
                    if line.strip():
                        predictions.append(json.loads(line))

        return predictions

    def append_retrieval_trace(self, run_id: str, trace: dict[str, Any]) -> None:
        """
        Append a retrieval trace to the traces file.

        Args:
            run_id: Run identifier
            trace: Retrieval trace data
        """
        run_dir = self.create_run_dir(run_id)
        traces_path = run_dir / "retrieval_traces.jsonl"

        with open(traces_path, "a") as f:
            f.write(json.dumps(trace) + "\n")

    def save_retrieval_traces(self, run_id: str, traces: list[dict[str, Any]]) -> Path:
        """
        Save all retrieval traces to JSONL file.

        Args:
            run_id: Run identifier
            traces: List of trace dictionaries

        Returns:
            Path to saved traces file
        """
        run_dir = self.create_run_dir(run_id)
        traces_path = run_dir / "retrieval_traces.jsonl"

        with open(traces_path, "w") as f:
            for trace in traces:
                f.write(json.dumps(trace) + "\n")

        logger.debug(f"Saved {len(traces)} retrieval traces to {traces_path}")
        return traces_path

    def load_retrieval_traces(self, run_id: str) -> list[dict[str, Any]]:
        """Load retrieval traces from JSONL file."""
        traces_path = self.get_run_dir(run_id) / "retrieval_traces.jsonl"
        traces = []

        if traces_path.exists():
            with open(traces_path) as f:
                for line in f:
                    if line.strip():
                        traces.append(json.loads(line))

        return traces

    def save_metrics(self, run_id: str, metrics: list[dict[str, Any]]) -> Path:
        """
        Save per-example metrics to JSONL file.

        Args:
            run_id: Run identifier
            metrics: List of metric dictionaries

        Returns:
            Path to saved metrics file
        """
        run_dir = self.create_run_dir(run_id)
        metrics_path = run_dir / "metrics.jsonl"

        with open(metrics_path, "w") as f:
            for metric in metrics:
                f.write(json.dumps(metric) + "\n")

        logger.debug(f"Saved {len(metrics)} metrics to {metrics_path}")
        return metrics_path

    def load_metrics(self, run_id: str) -> list[dict[str, Any]]:
        """Load metrics from JSONL file."""
        metrics_path = self.get_run_dir(run_id) / "metrics.jsonl"
        metrics = []

        if metrics_path.exists():
            with open(metrics_path) as f:
                for line in f:
                    if line.strip():
                        metrics.append(json.loads(line))

        return metrics

    def save_report(self, run_id: str, report_name: str, content: str) -> Path:
        """
        Save a markdown report.

        Args:
            run_id: Run identifier
            report_name: Name of the report (e.g., "summary", "attribution")
            content: Markdown content

        Returns:
            Path to saved report
        """
        run_dir = self.create_run_dir(run_id)
        report_path = run_dir / "reports" / f"{report_name}.md"

        with open(report_path, "w") as f:
            f.write(content)

        logger.debug(f"Saved report to {report_path}")
        return report_path

    def load_report(self, run_id: str, report_name: str) -> str | None:
        """Load a markdown report."""
        report_path = self.get_run_dir(run_id) / "reports" / f"{report_name}.md"

        if report_path.exists():
            with open(report_path) as f:
                return f.read()
        return None

    def save_json(self, run_id: str, filename: str, data: Any) -> Path:
        """
        Save arbitrary JSON data.

        Args:
            run_id: Run identifier
            filename: Output filename
            data: Data to serialize

        Returns:
            Path to saved file
        """
        run_dir = self.create_run_dir(run_id)
        file_path = run_dir / filename

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

        return file_path

    def load_json(self, run_id: str, filename: str) -> Any:
        """Load JSON data."""
        file_path = self.get_run_dir(run_id) / filename

        with open(file_path) as f:
            return json.load(f)

    def delete_run(self, run_id: str) -> bool:
        """
        Delete all artifacts for a run.

        Args:
            run_id: Run identifier

        Returns:
            True if deleted, False if not found
        """
        run_dir = self.get_run_dir(run_id)

        if run_dir.exists():
            shutil.rmtree(run_dir)
            logger.info(f"Deleted artifacts for run {run_id}")
            return True
        return False

    def list_runs(self) -> list[str]:
        """
        List all run IDs with stored artifacts.

        Returns:
            List of run IDs
        """
        runs = []
        if self.base_dir.exists():
            for item in self.base_dir.iterdir():
                if item.is_dir() and not item.name.startswith("."):
                    runs.append(item.name)
        return sorted(runs)

    def get_run_info(self, run_id: str) -> dict[str, Any] | None:
        """
        Get summary info about a run's artifacts.

        Args:
            run_id: Run identifier

        Returns:
            Dictionary with artifact info, or None if not found
        """
        run_dir = self.get_run_dir(run_id)

        if not run_dir.exists():
            return None

        info = {
            "run_id": run_id,
            "path": str(run_dir),
            "files": [],
        }

        for item in run_dir.rglob("*"):
            if item.is_file():
                rel_path = item.relative_to(run_dir)
                info["files"].append(
                    {
                        "path": str(rel_path),
                        "size_bytes": item.stat().st_size,
                        "modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat(),
                    }
                )

        return info
