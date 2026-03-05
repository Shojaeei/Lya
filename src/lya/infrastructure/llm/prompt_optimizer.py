"""Prompt Optimizer for Lya.

Implements DSPy-style prompt optimization and self-improvement.
Optimizes prompts through evaluation and iteration.
Pure Python 3.14+ compatible.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Example:
    """Training example for prompt optimization."""
    input: str
    expected_output: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptSignature:
    """Signature defining inputs and outputs."""
    name: str
    inputs: list[str]
    outputs: list[str]
    instructions: str


@dataclass
class PromptTemplate:
    """A prompt template with optimization history."""
    signature: PromptSignature
    template: str
    version: int = 1
    score: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    history: list[dict] = field(default_factory=list)

    def format(self, **kwargs) -> str:
        """Format the template with values."""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            return f"[Error: Missing variable {e}]"


@dataclass
class EvaluationResult:
    """Result of prompt evaluation."""
    score: float
    metrics: dict[str, float]
    errors: list[str]
    examples_passed: int
    examples_failed: int


class PromptOptimizer:
    """
    Optimizes prompts through iterative improvement.

    Inspired by DSPy, uses few-shot examples and metric-based optimization.
    """

    def __init__(
        self,
        llm_client: Any | None = None,
        max_iterations: int = 10,
        examples_per_prompt: int = 5,
    ) -> None:
        """Initialize optimizer.

        Args:
            llm_client: LLM client for generation
            max_iterations: Maximum optimization iterations
            examples_per_prompt: Number of examples to include
        """
        self.llm = llm_client
        self.max_iterations = max_iterations
        self.examples_per_prompt = examples_per_prompt

        self._examples: list[Example] = []
        self._templates: dict[str, list[PromptTemplate]] = {}

    def add_examples(self, examples: list[Example]) -> None:
        """Add training examples.

        Args:
            examples: Examples for optimization
        """
        self._examples.extend(examples)

    def create_signature(
        self,
        name: str,
        inputs: list[str],
        outputs: list[str],
        instructions: str,
    ) -> PromptSignature:
        """Create a prompt signature.

        Args:
            name: Signature name
            inputs: Input field names
            outputs: Output field names
            instructions: Base instructions

        Returns:
            Prompt signature
        """
        return PromptSignature(
            name=name,
            inputs=inputs,
            outputs=outputs,
            instructions=instructions,
        )

    def create_initial_template(self, signature: PromptSignature) -> PromptTemplate:
        """Create initial prompt template from signature.

        Args:
            signature: Prompt signature

        Returns:
            Initial template
        """
        # Build template
        lines = [
            signature.instructions,
            "",
            "Inputs:",
        ]

        for inp in signature.inputs:
            lines.append(f"  {inp}: {{{inp}}}")

        lines.extend([
            "",
            "Provide your response in JSON format with these fields:",
        ])

        for out in signature.outputs:
            lines.append(f"  {out}: ...")

        lines.extend([
            "",
            "Response:",
        ])

        template = "\n".join(lines)

        return PromptTemplate(
            signature=signature,
            template=template,
        )

    def optimize(
        self,
        signature: PromptSignature,
        metric: Callable[[str, str], float],
        iterations: int | None = None,
    ) -> PromptTemplate:
        """Optimize a prompt template.

        Args:
            signature: Prompt signature
            metric: Evaluation metric function
            iterations: Number of iterations

        Returns:
            Optimized template
        """
        iterations = iterations or self.max_iterations

        # Start with initial template
        current = self.create_initial_template(signature)
        best = current
        best_score = 0.0

        for i in range(iterations):
            # Generate candidate
            candidate = self._generate_candidate(current, i)

            # Evaluate
            result = self._evaluate_template(candidate, metric)
            candidate.score = result.score

            # Track best
            if result.score > best_score:
                best = candidate
                best_score = result.score

            # Update current if improved
            if result.score > current.score:
                current = candidate

            # Store in history
            if signature.name not in self._templates:
                self._templates[signature.name] = []
            self._templates[signature.name].append(candidate)

        return best

    def _generate_candidate(
        self,
        current: PromptTemplate,
        iteration: int,
    ) -> PromptTemplate:
        """Generate candidate template variation."""
        template = current.template

        # Select examples
        examples = self._select_examples()

        # Add few-shot examples if available
        if examples and iteration > 0:
            example_section = self._format_examples(examples)
            template = self._insert_examples(template, example_section)

        # Apply mutations
        template = self._mutate_template(template, iteration)

        return PromptTemplate(
            signature=current.signature,
            template=template,
            version=current.version + 1,
            history=[{"parent": current.version, "mutation": iteration}],
        )

    def _select_examples(self) -> list[Example]:
        """Select examples for few-shot prompting."""
        if not self._examples:
            return []

        # Select diverse examples
        if len(self._examples) <= self.examples_per_prompt:
            return self._examples

        return random.sample(self._examples, self.examples_per_prompt)

    def _format_examples(self, examples: list[Example]) -> str:
        """Format examples for prompt."""
        lines = ["\nExamples:\n"]

        for i, ex in enumerate(examples, 1):
            lines.append(f"Example {i}:")
            lines.append(f"  Input: {ex.input}")
            lines.append(f"  Output: {ex.expected_output}")
            lines.append("")

        return "\n".join(lines)

    def _insert_examples(self, template: str, examples: str) -> str:
        """Insert examples into template."""
        # Insert before "Response:"
        if "Response:" in template:
            parts = template.rsplit("Response:", 1)
            return parts[0] + examples + "\nResponse:" + parts[1]
        return template + examples

    def _mutate_template(self, template: str, iteration: int) -> str:
        """Apply mutations to template."""
        mutations = [
            self._add_clarity,
            self._add_structure,
            self._add_constraints,
        ]

        # Apply mutation based on iteration
        mutation = mutations[iteration % len(mutations)]
        return mutation(template)

    def _add_clarity(self, template: str) -> str:
        """Add clarity instructions."""
        addition = "\n\nBe clear and concise in your response."
        if addition not in template:
            template += addition
        return template

    def _add_structure(self, template: str) -> str:
        """Add structure instructions."""
        addition = "\n\nStructure your response logically."
        if addition not in template:
            template += addition
        return template

    def _add_constraints(self, template: str) -> str:
        """Add constraint instructions."""
        addition = "\n\nEnsure your response is accurate and complete."
        if addition not in template:
            template += addition
        return template

    def _evaluate_template(
        self,
        template: PromptTemplate,
        metric: Callable[[str, str], float],
    ) -> EvaluationResult:
        """Evaluate template quality."""
        if not self._examples or not self.llm:
            return EvaluationResult(
                score=0.5,
                metrics={},
                errors=[],
                examples_passed=0,
                examples_failed=0,
            )

        scores = []
        errors = []
        passed = 0
        failed = 0

        for example in self._examples[:5]:  # Test on subset
            try:
                # Format prompt
                prompt = template.format(input=example.input)

                # Generate response (mock for now)
                response = f"Mock response for: {example.input}"

                # Score
                score = metric(response, example.expected_output)
                scores.append(score)

                if score > 0.7:
                    passed += 1
                else:
                    failed += 1

            except Exception as e:
                errors.append(str(e))
                failed += 1

        avg_score = sum(scores) / len(scores) if scores else 0.0

        return EvaluationResult(
            score=avg_score,
            metrics={"accuracy": avg_score},
            errors=errors,
            examples_passed=passed,
            examples_failed=failed,
        )

    def save(self, path: str | Path) -> None:
        """Save optimized templates."""
        data = {
            name: [
                {
                    "signature": asdict(t.signature),
                    "template": t.template,
                    "version": t.version,
                    "score": t.score,
                    "created_at": t.created_at,
                }
                for t in templates
            ]
            for name, templates in self._templates.items()
        }

        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: str | Path) -> None:
        """Load optimized templates."""
        data = json.loads(Path(path).read_text())

        for name, templates in data.items():
            self._templates[name] = [
                PromptTemplate(
                    signature=PromptSignature(**t["signature"]),
                    template=t["template"],
                    version=t["version"],
                    score=t["score"],
                    created_at=t["created_at"],
                )
                for t in templates
            ]

    def get_best_template(self, name: str) -> PromptTemplate | None:
        """Get best template for a signature."""
        templates = self._templates.get(name, [])
        if not templates:
            return None
        return max(templates, key=lambda t: t.score)


class BootstrapFewShot:
    """
    Bootstrap few-shot learning for prompts.

    Automatically selects and optimizes few-shot examples.
    """

    def __init__(
        self,
        metric: Callable[[str, str], float],
        max_bootstrapped: int = 5,
    ) -> None:
        """Initialize bootstrap few-shot.

        Args:
            metric: Evaluation metric
            max_bootstrapped: Max bootstrapped examples
        """
        self.metric = metric
        self.max_bootstrapped = max_bootstrapped
        self._demonstrations: list[Example] = []

    def compile(
        self,
        student: PromptTemplate,
        trainset: list[Example],
    ) -> PromptTemplate:
        """Compile optimized prompt with demonstrations.

        Args:
            student: Base prompt template
            trainset: Training examples

        Returns:
            Optimized template with demonstrations
        """
        # Score all examples
        scored_examples = []

        for example in trainset:
            score = self.metric(example.expected_output, example.expected_output)
            scored_examples.append((example, score))

        # Sort by score
        scored_examples.sort(key=lambda x: x[1], reverse=True)

        # Select top examples
        selected = scored_examples[:self.max_bootstrapped]

        # Build demonstration section
        demo_section = self._build_demonstrations([e for e, _ in selected])

        # Create new template with demonstrations
        new_template = self._insert_demonstrations(student.template, demo_section)

        return PromptTemplate(
            signature=student.signature,
            template=new_template,
            version=student.version + 1,
        )

    def _build_demonstrations(self, examples: list[Example]) -> str:
        """Build demonstration section."""
        lines = ["\n\nHere are some examples:\n"]

        for i, ex in enumerate(examples, 1):
            lines.append(f"Example {i}:")
            lines.append(f"  Input: {ex.input}")
            lines.append(f"  Output: {ex.expected_output}")
            lines.append("")

        return "\n".join(lines)

    def _insert_demonstrations(self, template: str, demos: str) -> str:
        """Insert demonstrations into template."""
        # Find insertion point (before final instruction)
        lines = template.split("\n")

        # Insert before "Response:" or at end
        for i, line in enumerate(lines):
            if "Response:" in line:
                lines.insert(i, demos)
                return "\n".join(lines)

        # Add at end
        return template + demos + "\n\nResponse:\n"


class ChainOfThought:
    """
    Chain-of-thought prompting module.

    Adds reasoning steps to prompts.
    """

    def __init__(self) -> None:
        """Initialize CoT module."""
        self.reasoning_template = """
Let's work through this step by step:

1. Understand the problem: {problem}
2. Identify key information: {information}
3. Apply reasoning: {reasoning}
4. Formulate answer: {answer}

Final answer: """

    def apply(self, template: PromptTemplate) -> PromptTemplate:
        """Apply chain-of-thought to template.

        Args:
            template: Base template

        Returns:
            Template with CoT reasoning
        """
        # Add reasoning instruction
        new_template = template.template + "\n\nThink through this step by step."

        return PromptTemplate(
            signature=template.signature,
            template=new_template,
            version=template.version + 1,
        )


# ═══════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════

def exact_match_metric(prediction: str, expected: str) -> float:
    """Exact match metric."""
    return 1.0 if prediction.strip() == expected.strip() else 0.0


def contains_metric(prediction: str, expected: str) -> float:
    """Check if prediction contains expected."""
    expected_clean = expected.strip().lower()
    prediction_clean = prediction.strip().lower()

    if expected_clean in prediction_clean:
        return 1.0

    # Partial credit for word overlap
    expected_words = set(expected_clean.split())
    prediction_words = set(prediction_clean.split())

    if not expected_words:
        return 0.0

    overlap = len(expected_words & prediction_words)
    return overlap / len(expected_words)


def fuzzy_match_metric(prediction: str, expected: str) -> float:
    """Fuzzy string matching metric."""
    from difflib import SequenceMatcher

    return SequenceMatcher(None, prediction.lower(), expected.lower()).ratio()


# ═══════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Create optimizer
    optimizer = PromptOptimizer()

    # Create signature
    signature = optimizer.create_signature(
        name="summarize",
        inputs=["text"],
        outputs=["summary"],
        instructions="Summarize the given text concisely.",
    )

    # Add training examples
    examples = [
        Example(
            input="The quick brown fox jumps over the lazy dog.",
            expected_output="A fox jumps over a dog.",
        ),
        Example(
            input="Python is a popular programming language.",
            expected_output="Python is popular.",
        ),
    ]
    optimizer.add_examples(examples)

    # Create initial template
    template = optimizer.create_initial_template(signature)
    print("Initial template:")
    print(template.template)
    print()

    # Optimize
    optimized = optimizer.optimize(signature, contains_metric, iterations=3)
    print("Optimized template:")
    print(optimized.template)
    print(f"\nScore: {optimized.score}")

    # Test with bootstrap few-shot
    bootstrap = BootstrapFewShot(metric=exact_match_metric)
    compiled = bootstrap.compile(template, examples)

    print("\nCompiled with demonstrations:")
    print(compiled.template)
