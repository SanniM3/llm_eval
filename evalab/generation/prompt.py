"""Jinja2 prompt template rendering."""

import logging
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, Template, TemplateNotFound

logger = logging.getLogger(__name__)


class PromptRenderer:
    """
    Renders prompts from Jinja2 templates.

    Supports loading templates from files or strings.
    """

    def __init__(
        self,
        template_dir: str | Path | None = None,
        autoescape: bool = False,
    ):
        """
        Initialize prompt renderer.

        Args:
            template_dir: Directory containing template files
            autoescape: Whether to autoescape HTML (usually False for prompts)
        """
        self.template_dir = Path(template_dir) if template_dir else None

        # Set up Jinja2 environment
        if self.template_dir and self.template_dir.exists():
            loader = FileSystemLoader(str(self.template_dir))
            self._env = Environment(loader=loader, autoescape=autoescape)
        else:
            self._env = Environment(autoescape=autoescape)

        # Cache for compiled templates
        self._template_cache: dict[str, Template] = {}

    def render(self, template_path: str, **variables: Any) -> str:
        """
        Render a template file with variables.

        Args:
            template_path: Path to template file (relative to template_dir)
            **variables: Variables to inject into template

        Returns:
            Rendered prompt string
        """
        try:
            template = self._env.get_template(template_path)
            return template.render(**variables)
        except TemplateNotFound:
            # Try as absolute path
            abs_path = Path(template_path)
            if abs_path.exists():
                return self.render_string(abs_path.read_text(), **variables)
            raise

    def render_string(self, template_string: str, **variables: Any) -> str:
        """
        Render a template string with variables.

        Args:
            template_string: Jinja2 template as string
            **variables: Variables to inject

        Returns:
            Rendered prompt string
        """
        # Check cache
        if template_string in self._template_cache:
            template = self._template_cache[template_string]
        else:
            template = self._env.from_string(template_string)
            self._template_cache[template_string] = template

        return template.render(**variables)

    def load_template(self, template_path: str | Path) -> str:
        """
        Load a template file content.

        Args:
            template_path: Path to template file

        Returns:
            Template content as string
        """
        path = Path(template_path)
        if not path.is_absolute() and self.template_dir:
            path = self.template_dir / path

        return path.read_text()

    def list_templates(self) -> list[str]:
        """
        List available template files.

        Returns:
            List of template filenames
        """
        if self.template_dir and self.template_dir.exists():
            return [f.name for f in self.template_dir.glob("*.jinja")]
        return []


# Default prompt templates
DEFAULT_TEMPLATES = {
    "qa": """Answer the following question based on the provided context.

Context:
{{ context }}

Question: {{ question }}

Answer:""",
    "rag_qa": """You are a helpful assistant. Answer the question based ONLY on the provided context.
If the answer cannot be found in the context, say "I don't have enough information to answer this question."

Context:
{% for chunk in chunks %}
[Document {{ loop.index }}]
{{ chunk.text }}
{% endfor %}

Question: {{ question }}

Provide a concise answer with citations to the relevant document numbers.""",
    "summarization": """Summarize the following document concisely.

Document:
{{ document }}

Summary:""",
    "classification": """Classify the following text into one of these categories: {{ labels | join(', ') }}

Text: {{ text }}

Category:""",
    "judge_faithfulness": """You are a faithfulness evaluator. Your task is to determine if each claim in the response is supported by the provided context.

Context:
{{ context }}

Response to evaluate:
{{ response }}

For each distinct claim in the response, determine if it is:
- SUPPORTED: The claim is directly supported by information in the context
- NOT_FOUND: The claim cannot be verified from the context (neither supported nor contradicted)
- CONTRADICTED: The claim directly contradicts information in the context

Provide your evaluation in JSON format:
{
    "claims": [
        {"claim": "...", "verdict": "SUPPORTED|NOT_FOUND|CONTRADICTED", "evidence": "..."}
    ],
    "overall_faithfulness": <float 0-1>
}""",
}


def get_default_template(task: str) -> str:
    """
    Get a default template for a task type.

    Args:
        task: Task type (qa, rag_qa, summarization, classification)

    Returns:
        Template string
    """
    if task not in DEFAULT_TEMPLATES:
        raise ValueError(f"No default template for task: {task}")
    return DEFAULT_TEMPLATES[task]
