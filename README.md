# LLM-EvalLab

**Production-Style Evaluation & Reliability Platform for LLM Applications**

LLM-EvalLab provides comprehensive evaluation capabilities for LLM-powered applications including QA, summarization, classification, and RAG systems. It offers dataset/prompt versioning, repeatable runs, multi-metric evaluation, component-wise error attribution, cost/latency tracking, and a dashboard + API for comparing runs and detecting regressions.

## Features

- **Multi-Task Support**: QA, RAG-QA, Summarization, Classification
- **Comprehensive Metrics**: Accuracy, Semantic Similarity, Faithfulness, Robustness, Calibration
- **RAG Evaluation**: BM25, Dense, and Hybrid retrieval with faithfulness grounding checks
- **Error Attribution**: Identify whether failures come from retrieval, generation, or prompts
- **Cost & Latency Tracking**: Token usage, latency distribution, cost estimation
- **Versioning**: Content-hash based dataset and config versioning for reproducibility
- **Multiple Interfaces**: CLI, REST API, and Streamlit Dashboard
- **Flexible Backends**: OpenAI API and vLLM for local GPU inference

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              LLM-EvalLab                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐ │
│  │   Dataset    │   │  Retrieval   │   │  Generation  │   │  Evaluation  │ │
│  │   Loader     │──▶│   System     │──▶│   Backend    │──▶│   Engine     │ │
│  └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘ │
│        │                   │                  │                   │         │
│        │            ┌──────┴──────┐    ┌─────┴─────┐      ┌──────┴──────┐  │
│        │            │ BM25/Dense/ │    │  OpenAI/  │      │  Accuracy/  │  │
│        │            │   Hybrid    │    │   vLLM    │      │  Semantic/  │  │
│        │            └─────────────┘    └───────────┘      │Faithfulness │  │
│        │                                                  └─────────────┘  │
│        ▼                                                         │         │
│  ┌──────────────┐                                               ▼         │
│  │   Storage    │◀──────────────────────────────────────────────┤         │
│  │  (SQLite +   │                                               │         │
│  │  Artifacts)  │                                     ┌─────────┴───────┐ │
│  └──────────────┘                                     │   Attribution   │ │
│        │                                              │    Analysis     │ │
│        ▼                                              └─────────────────┘ │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │                    CLI  │  REST API  │  Dashboard                    │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/llm-evallab.git
cd llm-evallab

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key"
```

### Initialize a Project

```bash
# Create project structure with sample configs and data
evalab init my_project
cd my_project
```

### Run Your First Evaluation

```bash
# Run evaluation with sample config
evalab run --config configs/sample_run.yaml
```

### View Results

```bash
# List all runs
evalab list-runs

# Generate a report
evalab report <run_id>

# Compare two runs
evalab compare <run_id_a> <run_id_b>

# Start the dashboard
evalab serve
```

## Docker

```bash
# Build and run with Docker Compose
docker-compose up -d

# Access:
# - API: http://localhost:8080
# - Dashboard: http://localhost:8501
```

## Configuration

Create a YAML configuration file:

```yaml
run_name: "my_evaluation"

dataset:
  path: "data/datasets/my_data.jsonl"

retrieval:
  enabled: true
  corpus_path: "data/corpus/documents.jsonl"
  retriever:
    type: "hybrid"
    bm25_weight: 0.4
    dense_weight: 0.6
  top_k: 5

generation:
  backend: "openai"
  model: "gpt-4o-mini"
  temperature: 0.2

prompt:
  template_path: "prompts/rag_qa_v1.jinja"

evaluation:
  suites:
    - "accuracy"
    - "semantic"
    - "faithfulness"
    - "cost_latency"

attribution:
  enabled: true
  probes:
    - "gold_context"
```

## Dataset Format

Datasets are JSONL files with one example per line:

```json
{
  "id": "qa_001",
  "task": "qa",
  "input": {"question": "What is the capital of France?", "context": "..."},
  "reference": {"answer": "Paris", "aliases": ["paris"]},
  "metadata": {"domain": "geography", "difficulty": "easy"}
}
```

## Evaluation Metrics

### Accuracy Suite
- **Exact Match (EM)**: Binary match after normalization
- **Token F1**: Token-level precision/recall/F1

### Semantic Suite
- **BERTScore**: Contextual embedding similarity
- **Sentence Similarity**: Cosine similarity of sentence embeddings

### Faithfulness Suite
- **Faithfulness Score**: Claims supported by retrieved context
- **Unsupported Claim Rate**: Proportion of ungrounded claims
- **Citation Precision**: Accuracy of document citations

### Robustness Suite
- **Output Consistency**: Semantic stability under perturbations
- **Decision Flip Rate**: Changed predictions after perturbations

### Calibration Suite
- **ECE**: Expected Calibration Error
- **Confidence-Accuracy Correlation**: How well confidence predicts correctness

### Cost/Latency Suite
- Token counts (input/output)
- Latency distribution (p50, p95, p99)
- Estimated cost in USD

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/runs` | Create a new run |
| GET | `/runs` | List all runs |
| GET | `/runs/{id}` | Get run details |
| GET | `/runs/{id}/metrics` | Get run metrics |
| GET | `/runs/{id}/examples/{ex_id}` | Get example details |
| GET | `/compare?a=...&b=...` | Compare two runs |
| DELETE | `/runs/{id}` | Delete a run |

## Adding a New Metric

1. Create a new suite in `evalab/evaluation/`:

```python
from evalab.evaluation.base import EvaluationSuite, MetricResult

class MyCustomSuite(EvaluationSuite):
    def __init__(self):
        super().__init__(name="my_custom")

    def evaluate(self, prediction, reference, **kwargs):
        # Your metric logic
        score = compute_my_metric(prediction, reference)
        return [MetricResult(name="my_metric", value=score)]
```

2. Register it in `evalab/evaluation/__init__.py`

## Adding a New LLM Backend

1. Implement the `LLMBackend` interface:

```python
from evalab.generation.base import LLMBackend
from evalab.generation.result import GenerationResult

class MyBackend(LLMBackend):
    def generate(self, prompt, **kwargs) -> GenerationResult:
        # Your generation logic
        return GenerationResult(
            text=output,
            input_tokens=n_in,
            output_tokens=n_out,
            latency_ms=latency,
        )

    def count_tokens(self, text) -> int:
        return len(text) // 4  # Approximate
```

2. Add to `evalab/generation/vllm_backend.py` factory function

## Project Structure

```
llm_eval/
├── evalab/                 # Main package
│   ├── config/            # Configuration schemas
│   ├── data/              # Data loading & validation
│   ├── retrieval/         # BM25, Dense, Hybrid retrievers
│   ├── generation/        # LLM backends
│   ├── evaluation/        # Metric suites
│   ├── attribution/       # Error analysis
│   ├── storage/           # Database & artifacts
│   ├── pipeline/          # Orchestration
│   ├── api/               # FastAPI endpoints
│   ├── dashboard/         # Streamlit UI
│   └── cli.py             # CLI commands
├── prompts/               # Jinja2 templates
├── configs/               # Sample configurations
├── data/                  # Sample datasets
├── tests/                 # Test suite
├── Dockerfile
└── docker-compose.yml
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=evalab

# Run specific test file
pytest tests/unit/test_metrics.py -v
```

## Limitations & Future Work

**Current Limitations:**
- Single-node execution (no distributed evaluation)
- English-only for some semantic metrics
- No human annotation interface (minimal support only)

**Planned Improvements:**
- Distributed evaluation across multiple workers
- Additional LLM backends (Anthropic, Cohere, local models)
- Human-in-the-loop annotation workflow
- Automatic regression detection and alerting
- Export to W&B / MLflow

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please read CONTRIBUTING.md for guidelines.
