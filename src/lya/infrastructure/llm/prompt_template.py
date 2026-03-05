"""Prompt templating system."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lya.infrastructure.config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PromptTemplate:
    """Template for LLM prompts."""

    name: str
    template: str
    description: str = ""
    variables: list[str] = field(default_factory=list)
    system_prompt: str | None = None

    def __post_init__(self):
        """Extract variables from template if not provided."""
        if not self.variables:
            # Find {{variable}} patterns
            self.variables = re.findall(r'\{\{(\w+)\}\}', self.template)

    def render(self, **kwargs) -> str:
        """
        Render template with variables.

        Args:
            **kwargs: Variable values

        Returns:
            Rendered prompt
        """
        result = self.template

        for var in self.variables:
            value = kwargs.get(var, f"{{{{{var}}}}")
            result = result.replace(f"{{{{{var}}}}}", str(value))

        return result

    def render_with_system(self, **kwargs) -> tuple[str | None, str]:
        """
        Render template and return system/user prompts.

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        user_prompt = self.render(**kwargs)
        return self.system_prompt, user_prompt


class PromptManager:
    """Manager for prompt templates."""

    # Default templates
    DEFAULT_TEMPLATES: dict[str, PromptTemplate] = {
        "planning": PromptTemplate(
            name="planning",
            system_prompt="""You are a task planning assistant. Break down goals into clear,
actionable steps. Consider dependencies and potential challenges.""",
            template="""Goal: {{goal}}

Context:
{{context}}

Create a detailed plan to achieve this goal. Include:
1. Main tasks and subtasks
2. Estimated effort for each
3. Dependencies between tasks
4. Potential risks and mitigations

Format your response as a structured list.""",
            description="Generate task plans from goals",
        ),

        "code_generation": PromptTemplate(
            name="code_generation",
            system_prompt="""You are an expert Python developer. Write clean, well-documented,
type-hinted code following best practices. Include docstrings and error handling.""",
            template="""Generate Python code for the following requirement:

{{requirement}}

Additional context:
{{context}}

Requirements:
- Follow PEP 8 style
- Include type hints
- Add docstrings
- Handle edge cases
- Make it modular and testable

Return only the code, no explanations.""",
            description="Generate Python code from requirements",
        ),

        "code_review": PromptTemplate(
            name="code_review",
            system_prompt="""You are a code reviewer. Analyze code for bugs, security issues,
performance problems, and style violations. Provide constructive feedback.""",
            template="""Review the following code:

```python
{{code}}
```

Context: {{context}}

Provide feedback on:
1. Bugs or logical errors
2. Security concerns
3. Performance issues
4. Style and readability
5. Suggestions for improvement

Format: List each issue with line numbers if applicable.""",
            description="Review code for issues",
        ),

        "summarize": PromptTemplate(
            name="summarize",
            system_prompt="You are a summarization assistant. Create concise summaries preserving key information.",
            template="""Summarize the following content:

{{content}}

Requirements:
- Be concise but comprehensive
- Preserve key facts and insights
- Use bullet points for clarity
- Maximum length: {{max_length}} words
""",
            description="Summarize text content",
        ),

        "conversation": PromptTemplate(
            name="conversation",
            system_prompt="""You are Lya, an autonomous AI agent. You are helpful, intelligent,
and capable of complex reasoning. You can use tools and capabilities to achieve goals.""",
            template="""{{message}}

Current context:
- Active goals: {{goals}}
- Current task: {{task}}
- Available capabilities: {{capabilities}}

Respond naturally and helpfully.""",
            description="Handle user conversation",
        ),

        "reflect": PromptTemplate(
            name="reflect",
            system_prompt="""You are a self-reflection assistant. Analyze experiences and extract
learnings, patterns, and insights for improvement.""",
            template="""Reflect on the following experience:

{{experience}}

Consider:
1. What went well?
2. What could be improved?
3. What patterns do you notice?
4. What would you do differently?
5. What have you learned?

Format as a structured reflection.""",
            description="Generate reflections on experiences",
        ),

        "decision": PromptTemplate(
            name="decision",
            system_prompt="You are a decision-making assistant. Analyze options and recommend the best course of action.",
            template="""Decision to make: {{question}}

Options:
{{options}}

Criteria:
{{criteria}}

Analyze each option against the criteria and recommend the best choice with reasoning.""",
            description="Analyze and make decisions",
        ),

        "error_analysis": PromptTemplate(
            name="error_analysis",
            system_prompt="You are a debugging assistant. Analyze errors and provide solutions.",
            template="""Analyze the following error:

Error: {{error}}

Context:
{{context}}

Provide:
1. Root cause analysis
2. Potential fixes
3. Prevention strategies
4. Related documentation or resources""",
            description="Analyze errors and suggest fixes",
        ),
    }

    def __init__(self):
        self._templates: dict[str, PromptTemplate] = dict(self.DEFAULT_TEMPLATES)
        self._custom_templates_path: Path | None = None

    def get(self, name: str) -> PromptTemplate | None:
        """Get a template by name."""
        return self._templates.get(name)

    def register(self, template: PromptTemplate) -> None:
        """Register a new template."""
        self._templates[template.name] = template
        logger.debug("Registered prompt template", name=template.name)

    def register_from_file(self, path: Path) -> None:
        """Load and register templates from a file."""
        import json

        with open(path) as f:
            data = json.load(f)

        for name, template_data in data.items():
            template = PromptTemplate(
                name=name,
                template=template_data["template"],
                description=template_data.get("description", ""),
                system_prompt=template_data.get("system_prompt"),
                variables=template_data.get("variables", []),
            )
            self.register(template)

        logger.info("Loaded templates from file", path=str(path), count=len(data))

    def list_templates(self) -> list[str]:
        """List all available template names."""
        return list(self._templates.keys())

    def render(self, template_name: str, **kwargs) -> str:
        """
        Render a template by name.

        Args:
            template_name: Name of template
            **kwargs: Variable values

        Returns:
            Rendered prompt

        Raises:
            KeyError: If template not found
        """
        template = self._templates.get(template_name)
        if not template:
            raise KeyError(f"Template not found: {template_name}")
        return template.render(**kwargs)

    def render_with_system(self, template_name: str, **kwargs) -> tuple[str | None, str]:
        """
        Render template and return system/user prompts.

        Args:
            template_name: Name of template
            **kwargs: Variable values

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        template = self._templates.get(template_name)
        if not template:
            raise KeyError(f"Template not found: {template_name}")
        return template.render_with_system(**kwargs)


# Global prompt manager instance
_prompt_manager: PromptManager | None = None


def get_prompt_manager() -> PromptManager:
    """Get the global prompt manager."""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager


def render_prompt(template_name: str, **kwargs) -> str:
    """Convenience function to render a prompt."""
    return get_prompt_manager().render(template_name, **kwargs)
