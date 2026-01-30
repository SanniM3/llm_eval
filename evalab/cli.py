"""Command-line interface for LLM-EvalLab."""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="evalab",
    help="LLM-EvalLab: Evaluation & Reliability Platform for LLM Applications",
    add_completion=False,
)

console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@app.command()
def init(
    path: Path = typer.Argument(
        Path("."),
        help="Directory to initialize",
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Overwrite existing files",
    ),
) -> None:
    """Initialize a new LLM-EvalLab project with sample configs and data."""
    console.print("[bold blue]Initializing LLM-EvalLab project...[/bold blue]")

    # Create directories
    dirs = [
        path / "configs",
        path / "data" / "datasets",
        path / "data" / "corpus",
        path / "prompts",
        path / "runs",
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        console.print(f"  Created: {d}")

    # Create sample config
    sample_config = path / "configs" / "sample_run.yaml"
    if not sample_config.exists() or force:
        sample_config.write_text("""# Sample LLM-EvalLab Run Configuration
run_name: "sample_qa_run"

dataset:
  path: "data/datasets/sample_qa.jsonl"
  name: "sample_qa"

retrieval:
  enabled: false

generation:
  backend: "openai"
  model: "gpt-4o-mini"
  temperature: 0.2
  max_tokens: 512

prompt:
  template_path: "prompts/qa_v1.jinja"

evaluation:
  suites:
    - "accuracy"
    - "semantic"
    - "cost_latency"

attribution:
  enabled: false

logging:
  save_traces: true
  output_dir: "runs/"
""")
        console.print(f"  Created: {sample_config}")

    # Create sample dataset
    sample_dataset = path / "data" / "datasets" / "sample_qa.jsonl"
    if not sample_dataset.exists() or force:
        samples = [
            {
                "id": "qa_001",
                "task": "qa",
                "input": {"question": "What is the capital of France?", "context": "France is a country in Europe. Its capital city is Paris, which is known for the Eiffel Tower."},
                "reference": {"answer": "Paris", "aliases": ["paris"]},
                "metadata": {"domain": "geography", "difficulty": "easy"}
            },
            {
                "id": "qa_002",
                "task": "qa",
                "input": {"question": "What year did World War II end?", "context": "World War II was a global conflict that lasted from 1939 to 1945. It ended with the surrender of Japan in September 1945."},
                "reference": {"answer": "1945", "aliases": ["september 1945"]},
                "metadata": {"domain": "history", "difficulty": "easy"}
            },
            {
                "id": "qa_003",
                "task": "qa",
                "input": {"question": "What is photosynthesis?", "context": "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen."},
                "reference": {"answer": "The process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen"},
                "metadata": {"domain": "science", "difficulty": "medium"}
            },
        ]
        with open(sample_dataset, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        console.print(f"  Created: {sample_dataset}")

    # Create sample prompt template
    sample_prompt = path / "prompts" / "qa_v1.jinja"
    if not sample_prompt.exists() or force:
        sample_prompt.write_text("""Answer the following question based on the provided context. Be concise and accurate.

{% if context %}
Context:
{{ context }}

{% endif %}
Question: {{ question }}

Answer:""")
        console.print(f"  Created: {sample_prompt}")

    console.print("\n[bold green]✓ Project initialized successfully![/bold green]")
    console.print("\nNext steps:")
    console.print("  1. Set your OPENAI_API_KEY environment variable")
    console.print("  2. Run: evalab run --config configs/sample_run.yaml")


@app.command()
def run(
    config: Path = typer.Option(
        ..., "--config", "-c",
        help="Path to run configuration YAML file",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Enable verbose logging",
    ),
) -> None:
    """Execute an evaluation run from a configuration file."""
    setup_logging(verbose)

    console.print(f"[bold blue]Loading config from: {config}[/bold blue]")

    if not config.exists():
        console.print(f"[red]Error: Config file not found: {config}[/red]")
        raise typer.Exit(1)

    from evalab.config.schemas import RunConfig
    from evalab.pipeline.runner import PipelineRunner

    try:
        run_config = RunConfig.from_yaml(config)
        console.print(f"  Run name: {run_config.run_name}")
        console.print(f"  Model: {run_config.generation.model}")
        console.print(f"  Dataset: {run_config.dataset.path}")

        runner = PipelineRunner(run_config)

        console.print("\n[bold]Starting evaluation run...[/bold]")
        result = runner.run()

        if result.status == "completed":
            console.print(f"\n[bold green]✓ Run completed: {result.run_id}[/bold green]")
            console.print(f"  Examples processed: {result.num_examples}")
            console.print(f"  Duration: {result.duration_sec:.2f}s")

            # Show metrics summary
            if result.metrics_summary:
                console.print("\n[bold]Metrics Summary:[/bold]")
                table = Table(show_header=True)
                table.add_column("Metric")
                table.add_column("Mean", justify="right")
                table.add_column("Std", justify="right")

                for name, stats in result.metrics_summary.items():
                    table.add_row(
                        name,
                        f"{stats['mean']:.4f}",
                        f"{stats['std']:.4f}",
                    )

                console.print(table)
        else:
            console.print(f"\n[bold red]✗ Run failed: {result.error}[/bold red]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            import traceback
            traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def compare(
    run_a: str = typer.Argument(..., help="First run ID"),
    run_b: str = typer.Argument(..., help="Second run ID"),
    slices: Optional[str] = typer.Option(
        None, "--slices", "-s",
        help="Comma-separated slice keys (e.g., domain,difficulty)",
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output file for comparison report (markdown)",
    ),
) -> None:
    """Compare metrics between two runs."""
    from evalab.storage.database import init_db
    from evalab.storage.registry import RunRegistry

    init_db()
    registry = RunRegistry()

    console.print(f"[bold blue]Comparing runs:[/bold blue]")
    console.print(f"  Run A: {run_a}")
    console.print(f"  Run B: {run_b}")

    try:
        comparison = registry.compare_runs(run_a, run_b)

        # Display comparison table
        table = Table(show_header=True, title="Metric Comparison")
        table.add_column("Metric")
        table.add_column("Run A (Mean)", justify="right")
        table.add_column("Run B (Mean)", justify="right")
        table.add_column("Delta", justify="right")

        for metric_name, data in comparison["metrics"].items():
            a_mean = data.get("run_a", {}).get("mean", 0)
            b_mean = data.get("run_b", {}).get("mean", 0)
            delta = data.get("delta", {}).get("mean", 0)

            delta_str = f"{delta:+.4f}"
            if delta > 0:
                delta_str = f"[green]{delta_str}[/green]"
            elif delta < 0:
                delta_str = f"[red]{delta_str}[/red]"

            table.add_row(
                metric_name,
                f"{a_mean:.4f}" if a_mean else "-",
                f"{b_mean:.4f}" if b_mean else "-",
                delta_str,
            )

        console.print(table)

        # Save report if requested
        if output:
            report = f"# Run Comparison Report\n\n"
            report += f"**Run A:** {run_a}\n**Run B:** {run_b}\n\n"
            report += "## Metrics\n\n"
            report += "| Metric | Run A | Run B | Delta |\n"
            report += "|--------|-------|-------|-------|\n"

            for metric_name, data in comparison["metrics"].items():
                a_mean = data.get("run_a", {}).get("mean", 0)
                b_mean = data.get("run_b", {}).get("mean", 0)
                delta = data.get("delta", {}).get("mean", 0)
                report += f"| {metric_name} | {a_mean:.4f} | {b_mean:.4f} | {delta:+.4f} |\n"

            output.write_text(report)
            console.print(f"\n[green]Report saved to: {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def report(
    run_id: str = typer.Argument(..., help="Run ID to generate report for"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output file (default: stdout)",
    ),
) -> None:
    """Generate a human-readable report for a run."""
    from evalab.storage.database import init_db
    from evalab.storage.registry import RunRegistry

    init_db()
    registry = RunRegistry()

    try:
        run = registry.get_run(run_id)
        if not run:
            console.print(f"[red]Run not found: {run_id}[/red]")
            raise typer.Exit(1)

        aggregates = registry.get_aggregates(run_id)

        # Build report
        report_lines = [
            f"# Evaluation Report: {run.name}",
            f"",
            f"**Run ID:** {run.id}",
            f"**Created:** {run.created_at}",
            f"**Status:** {run.status}",
            f"",
            f"## Configuration",
            f"",
            f"- Model: {run.config_json.get('generation', {}).get('model', 'N/A')}",
            f"- Dataset: {run.config_json.get('dataset', {}).get('path', 'N/A')}",
            f"",
            f"## Metrics Summary",
            f"",
        ]

        if aggregates:
            report_lines.append("| Metric | Mean | Std | Min | Max |")
            report_lines.append("|--------|------|-----|-----|-----|")

            for agg in aggregates:
                if agg.slice_key is None:  # Global aggregates
                    stats = agg.agg_json
                    report_lines.append(
                        f"| {agg.metric_name} | {stats.get('mean', 0):.4f} | "
                        f"{stats.get('std', 0):.4f} | {stats.get('min', 0):.4f} | "
                        f"{stats.get('max', 0):.4f} |"
                    )
        else:
            report_lines.append("*No metrics available*")

        report_text = "\n".join(report_lines)

        if output:
            output.write_text(report_text)
            console.print(f"[green]Report saved to: {output}[/green]")
        else:
            console.print(report_text)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def list_runs(
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum runs to show"),
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Filter by status"),
) -> None:
    """List all evaluation runs."""
    from evalab.storage.database import init_db
    from evalab.storage.registry import RunRegistry

    init_db()
    registry = RunRegistry()

    runs = registry.list_runs(status=status, limit=limit)

    if not runs:
        console.print("[yellow]No runs found.[/yellow]")
        return

    table = Table(show_header=True, title="Evaluation Runs")
    table.add_column("Run ID")
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("Created")
    table.add_column("Model")

    for run in runs:
        status_str = run.status
        if status_str == "completed":
            status_str = f"[green]{status_str}[/green]"
        elif status_str == "failed":
            status_str = f"[red]{status_str}[/red]"
        elif status_str == "running":
            status_str = f"[yellow]{status_str}[/yellow]"

        model = run.config_json.get("generation", {}).get("model", "N/A")

        table.add_row(
            run.id[:30] + "..." if len(run.id) > 30 else run.id,
            run.name,
            status_str,
            run.created_at.strftime("%Y-%m-%d %H:%M"),
            model,
        )

    console.print(table)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8080, "--port", "-p", help="Port for API"),
    dashboard_port: int = typer.Option(8501, "--dashboard-port", "-d", help="Port for dashboard"),
    no_dashboard: bool = typer.Option(False, "--no-dashboard", help="Disable dashboard"),
) -> None:
    """Start the API server and dashboard."""
    import threading

    from evalab.api.main import create_app

    console.print("[bold blue]Starting LLM-EvalLab servers...[/bold blue]")

    # Start API
    console.print(f"  API: http://{host}:{port}")

    if not no_dashboard:
        console.print(f"  Dashboard: http://{host}:{dashboard_port}")

        # Start dashboard in background
        def run_dashboard():
            subprocess.run([
                sys.executable, "-m", "streamlit", "run",
                "evalab/dashboard/app.py",
                "--server.port", str(dashboard_port),
                "--server.address", host,
                "--server.headless", "true",
            ], check=False)

        dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
        dashboard_thread.start()

    # Start API (blocking)
    import uvicorn

    app_instance = create_app()
    uvicorn.run(app_instance, host=host, port=port)


@app.command()
def delete(
    run_id: str = typer.Argument(..., help="Run ID to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Delete a run and its artifacts."""
    from evalab.storage.database import init_db
    from evalab.storage.registry import RunRegistry

    init_db()
    registry = RunRegistry()

    if not force:
        confirm = typer.confirm(f"Are you sure you want to delete run {run_id}?")
        if not confirm:
            console.print("[yellow]Cancelled.[/yellow]")
            raise typer.Exit(0)

    if registry.delete_run(run_id):
        console.print(f"[green]Deleted run: {run_id}[/green]")
    else:
        console.print(f"[red]Run not found: {run_id}[/red]")
        raise typer.Exit(1)


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
