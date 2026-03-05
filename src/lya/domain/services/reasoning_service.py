"""Reasoning service for chain-of-thought and metacognition."""

from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import UUID


class LLMPort(Protocol):
    """Port for LLM interactions."""

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text using LLM."""
        ...


@dataclass
class ReasoningChain:
    """A chain of reasoning steps."""
    steps: list[dict[str, Any]] = field(default_factory=list)
    conclusion: str = ""
    confidence: float = 0.0

    def add_step(self, thought: str, evidence: str | None = None) -> None:
        """Add a reasoning step."""
        self.steps.append({
            "order": len(self.steps) + 1,
            "thought": thought,
            "evidence": evidence,
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "conclusion": self.conclusion,
            "confidence": self.confidence,
        }


class ReasoningService:
    """
    Domain service for reasoning and metacognition.

    Handles chain-of-thought reasoning, self-reflection,
    and decision justification.
    """

    def __init__(self, llm: LLMPort) -> None:
        self._llm = llm

    async def reason(
        self,
        question: str,
        context: str | None = None,
    ) -> ReasoningChain:
        """
        Perform chain-of-thought reasoning.

        Args:
            question: The question or problem to reason about
            context: Optional context information

        Returns:
            A reasoning chain with conclusion
        """
        prompt = f"""Think through this step-by-step:

{question}

Break down your reasoning into clear steps."""

        if context:
            prompt = f"Context:\n{context}\n\n{prompt}"

        try:
            response = await self._llm.generate(prompt, temperature=0.3)

            # Parse the response into steps
            chain = ReasoningChain()

            lines = response.strip().split("\n")
            current_step = ""

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Try to identify steps (numbered lists, "Step", etc.)
                if line[0].isdigit() or line.lower().startswith(("step", "reasoning")):
                    if current_step:
                        chain.add_step(current_step)
                    current_step = line
                elif line.lower().startswith(("conclusion", "therefore")):
                    if current_step:
                        chain.add_step(current_step)
                    chain.conclusion = line.split(":", 1)[1].strip() if ":" in line else line
                else:
                    current_step += " " + line

            if current_step and not chain.conclusion:
                chain.add_step(current_step)

            # If no conclusion extracted, use last step
            if not chain.conclusion and chain.steps:
                chain.conclusion = chain.steps[-1]["thought"]

            chain.confidence = 0.7  # Default confidence
            return chain

        except Exception:
            # Fallback
            chain = ReasoningChain()
            chain.add_step(f"Direct consideration of: {question}")
            chain.conclusion = "Unable to provide detailed reasoning"
            chain.confidence = 0.3
            return chain

    async def reflect_on_decision(
        self,
        decision: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Reflect on a decision made by the agent.

        Args:
            decision: The decision made
            context: Context of the decision

        Returns:
            Reflection with insights and potential improvements
        """
        prompt = f"""Reflect on this decision:

Decision: {decision}

Context:
{context}

Provide:
1. What was the reasoning behind this decision?
2. Were there alternative approaches?
3. What could be improved?
4. Any lessons learned?
"""

        try:
            response = await self._llm.generate(prompt, temperature=0.4)

            return {
                "decision": decision,
                "reflection": response,
                "timestamp": "now",
            }

        except Exception:
            return {
                "decision": decision,
                "reflection": "Unable to generate reflection",
                "timestamp": "now",
            }

    async def evaluate_goal_success(
        self,
        goal_description: str,
        result: Any,
        execution_log: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Evaluate the success of a goal execution.

        Args:
            goal_description: The goal that was pursued
            result: The final result
            execution_log: Log of execution steps

        Returns:
            Evaluation with success metrics
        """
        prompt = f"""Evaluate the success of this goal execution:

Goal: {goal_description}

Result: {result}

Provide:
1. Was the goal achieved? (Yes/No/Partially)
2. Success score (0-100)
3. What worked well?
4. What could be improved?
"""

        try:
            response = await self._llm.generate(prompt, temperature=0.3)

            # Parse simple evaluation
            score = 50  # Default middle score
            if "score" in response.lower():
                import re
                match = re.search(r'(\d+)', response)
                if match:
                    score = int(match.group(1))

            return {
                "goal": goal_description,
                "success_score": score,
                "evaluation": response,
                "achieved": score >= 70,
            }

        except Exception:
            return {
                "goal": goal_description,
                "success_score": 50,
                "evaluation": "Unable to evaluate",
                "achieved": None,
            }

    async def generate_insight(
        self,
        observations: list[str],
        pattern_type: str = "general",
    ) -> str:
        """
        Generate an insight from observations.

        Args:
            observations: List of observations
            pattern_type: Type of pattern to look for

        Returns:
            Generated insight
        """
        obs_str = "\n".join(f"- {o}" for o in observations)

        prompt = f"""Based on these observations about {pattern_type}, what insight can you generate?

Observations:
{obs_str}

Provide a concise insight or pattern."""

        try:
            return await self._llm.generate(prompt, temperature=0.5)
        except Exception:
            return "Unable to generate insight from observations"
