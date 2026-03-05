"""Self-Improvement Service.

This is the core service that enables Lya to generate, validate, and
integrate new capabilities autonomously.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

from lya.domain.models.capability import (
    Capability,
    CapabilityManifest,
    CapabilityInterface,
    CapabilityStatus,
    FunctionSignature,
    TestCase,
    SafetyConfig,
    ValidationResult,
    ValidationStatus,
)
from lya.domain.models.events import (
    CapabilityGenerated,
    CapabilityValidated,
    CapabilityRegistered,
    ToolExecuted,
)
from lya.domain.models.events import EventPublisher
from lya.infrastructure.config.logging import get_logger

logger = get_logger(__name__)


class LLMPort(Protocol):
    """Port for LLM interactions."""
    async def generate(self, prompt: str, **kwargs: Any) -> str: ...
    async def generate_structured(self, prompt: str, output_schema: dict[str, Any], **kwargs: Any) -> dict[str, Any]: ...


class SandboxPort(Protocol):
    """Port for code sandbox."""
    async def validate(self, capability: Capability) -> ValidationResult: ...


class RegistryPort(Protocol):
    """Port for capability registry."""
    async def register(self, capability: Capability) -> None: ...
    def list_capabilities(self) -> list[str]: ...
    def get(self, capability_id: str) -> Capability | None: ...


@dataclass
class CapabilityNeed:
    """Identified need for a new capability."""
    name: str
    description: str
    reason: str
    suggested_interface: CapabilityInterface
    suggested_dependencies: list[str]
    urgency: int  # 1-5


@dataclass
class ResearchResult:
    """Result of researching best practices."""
    best_libraries: list[dict[str, Any]]
    implementation_approach: str
    security_considerations: list[str]
    testing_strategy: str
    code_patterns: list[str]


class SelfImprovementService:
    """
    Core service for autonomous capability generation.

    This service enables Lya to:
    1. Identify missing capabilities
    2. Research best practices
    3. Generate code with tests
    4. Validate in sandbox
    5. Register and integrate
    6. Monitor and improve over time
    """

    def __init__(
        self,
        llm: LLMPort,
        sandbox: SandboxPort,
        registry: RegistryPort,
        event_bus: EventPublisher,
        workspace: Path,
    ):
        self._llm = llm
        self._sandbox = sandbox
        self._registry = registry
        self._events = event_bus
        self._workspace = workspace
        self._generations_dir = workspace / "generations"
        self._generations_dir.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: Need Detection
    # ═══════════════════════════════════════════════════════════════

    async def identify_capability_gap(
        self,
        goal_description: str,
        context: str | None = None,
    ) -> CapabilityNeed | None:
        """
        Analyze a goal and identify if a new capability is needed.

        Args:
            goal_description: The goal to achieve
            context: Additional context

        Returns:
            CapabilityNeed if a new capability should be generated, None otherwise
        """
        # Get list of existing capabilities
        existing = self._registry.list_capabilities()

        prompt = f"""Analyze this goal and determine if a new capability is needed.

Goal: {goal_description}

Existing capabilities:
{chr(10).join(f"- {cap}" for cap in existing[:20])}

Context: {context or "None"}

Question: Can this goal be achieved with existing capabilities, or is a new tool needed?

If a new capability is needed, provide:
1. Name (descriptive, like "web_scraper" or "pdf_reader")
2. Description (what it does)
3. Why it's needed
4. Main function signature (name, parameters, return type)
5. Dependencies (Python packages needed)
6. Urgency (1-5, 5 being critical)

Output as JSON:
{{
  "new_capability_needed": true/false,
  "name": "capability_name",
  "description": "what it does",
  "reason": "why needed",
  "function": {{
    "name": "function_name",
    "parameters": {{"param1": "type", "param2": "type"}},
    "returns": "return_type"
  }},
  "dependencies": ["package1", "package2"],
  "urgency": 3
}}

Be conservative - only suggest new capability if truly needed."""

        schema = {
            "type": "object",
            "properties": {
                "new_capability_needed": {"type": "boolean"},
                "name": {"type": "string"},
                "description": {"type": "string"},
                "reason": {"type": "string"},
                "function": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "parameters": {"type": "object"},
                        "returns": {"type": "string"},
                    },
                },
                "dependencies": {"type": "array", "items": {"type": "string"}},
                "urgency": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "required": ["new_capability_needed"],
        }

        try:
            result = await self._llm.generate_structured(prompt, schema, temperature=0.3)

            if not result.get("new_capability_needed"):
                logger.info("No new capability needed", goal=goal_description[:50])
                return None

            func_data = result.get("function", {})
            interface = CapabilityInterface(
                functions=[
                    FunctionSignature(
                        name=func_data.get("name", "execute"),
                        description=result.get("description", ""),
                        parameters=func_data.get("parameters", {}),
                        returns={"type": func_data.get("returns", "Any")},
                        async_=True,
                    )
                ]
            )

            return CapabilityNeed(
                name=result["name"],
                description=result["description"],
                reason=result["reason"],
                suggested_interface=interface,
                suggested_dependencies=result.get("dependencies", []),
                urgency=result.get("urgency", 3),
            )

        except Exception as e:
            logger.error("Failed to identify capability gap", error=str(e))
            return None

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: Research
    # ═══════════════════════════════════════════════════════════════

    async def research_best_practices(self, need: CapabilityNeed) -> ResearchResult:
        """
        Research best practices for implementing the capability.

        Args:
            need: The capability need

        Returns:
            Research results with recommendations
        """
        prompt = f"""Research the best way to implement this capability:

Name: {need.name}
Description: {need.description}
Interface: {need.suggested_interface.to_dict()}
Suggested dependencies: {need.suggested_dependencies}

Provide research on:
1. Best Python libraries for this task (compare top 3 options)
2. Recommended implementation approach
3. Security considerations and pitfalls
4. Testing strategy
5. Code patterns and best practices

Output as JSON:
{{
  "best_libraries": [
    {{"name": "library_name", "pros": ["..."], "cons": ["..."], "recommended": true/false}}
  ],
  "implementation_approach": "step by step approach",
  "security_considerations": ["list of concerns"],
  "testing_strategy": "how to test this",
  "code_patterns": ["pattern1", "pattern2"]
}}"""

        schema = {
            "type": "object",
            "properties": {
                "best_libraries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "pros": {"type": "array", "items": {"type": "string"}},
                            "cons": {"type": "array", "items": {"type": "string"}},
                            "recommended": {"type": "boolean"},
                        },
                    },
                },
                "implementation_approach": {"type": "string"},
                "security_considerations": {"type": "array", "items": {"type": "string"}},
                "testing_strategy": {"type": "string"},
                "code_patterns": {"type": "array", "items": {"type": "string"}},
            },
        }

        try:
            result = await self._llm.generate_structured(prompt, schema, temperature=0.4)

            return ResearchResult(
                best_libraries=result.get("best_libraries", []),
                implementation_approach=result.get("implementation_approach", ""),
                security_considerations=result.get("security_considerations", []),
                testing_strategy=result.get("testing_strategy", ""),
                code_patterns=result.get("code_patterns", []),
            )

        except Exception as e:
            logger.error("Research failed", error=str(e))
            # Return default research
            return ResearchResult(
                best_libraries=[{"name": "standard_library", "pros": ["No dependencies"], "cons": ["Manual implementation"], "recommended": True}],
                implementation_approach="Use standard library with careful error handling",
                security_considerations=["Validate all inputs"],
                testing_strategy="Unit tests with mock data",
                code_patterns=["async/await", "try/except"],
            )

    # ═══════════════════════════════════════════════════════════════
    # Phase 3: Code Generation
    # ═══════════════════════════════════════════════════════════════

    async def generate_capability(
        self,
        need: CapabilityNeed,
        research: ResearchResult,
    ) -> Capability:
        """
        Generate capability code based on need and research.

        Args:
            need: The capability need
            research: Research results

        Returns:
            Generated capability (not yet validated)
        """
        logger.info("Generating capability", name=need.name, urgency=need.urgency)

        # Select best library
        recommended = next(
            (lib for lib in research.best_libraries if lib.get("recommended")),
            research.best_libraries[0] if research.best_libraries else None
        )

        # Generate main code
        code_prompt = f"""Generate a complete Python module for this capability.

Capability: {need.name}
Description: {need.description}

Interface:
{json.dumps(need.suggested_interface.to_dict(), indent=2)}

Implementation approach:
{research.implementation_approach}

Recommended library: {recommended['name'] if recommended else 'standard library'}
Security considerations:
{chr(10).join(f"- {s}" for s in research.security_considerations)}

Requirements:
1. All functions must be async
2. Include type hints for all parameters and returns
3. Include complete docstrings (Google style)
4. Handle all errors gracefully with try/except
5. Follow PEP 8
6. Maximum 100 lines per function
7. Use the recommended library if applicable
8. Include input validation
9. Log important events
10. Do NOT include test code

The module should be complete and runnable.

Output ONLY the Python code, no markdown, no explanations."""

        try:
            code = await self._llm.generate(code_prompt, temperature=0.2)

            # Clean up code (remove markdown if present)
            code = self._clean_code(code)

        except Exception as e:
            logger.error("Code generation failed", error=str(e))
            code = self._generate_stub_code(need)

        # Generate tests
        test_code = await self._generate_tests(need, code, research)

        # Create manifest
        manifest = CapabilityManifest(
            id=f"{need.name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            name=need.name,
            version="1.0.0",
            description=need.description,
            author="lya_self",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            dependencies=need.suggested_dependencies,
            interface=need.suggested_interface,
            tests=[],  # Will be populated from test_code
            safety=SafetyConfig(
                sandbox=True,
                network_access=need.name in ["web_scraper", "api_client"],
                timeout_seconds=30,
            ),
        )

        capability = Capability(
            manifest=manifest,
            code=code,
            test_code=test_code,
            status=CapabilityStatus.DRAFT,
        )

        # Publish event
        await self._events.publish(
            CapabilityGenerated(
                capability_id=manifest.id,
                name=manifest.name,
                description=manifest.description,
            )
        )

        logger.info("Capability generated", id=manifest.id, name=manifest.name)
        return capability

    def _clean_code(self, code: str) -> str:
        """Clean generated code."""
        # Remove markdown code blocks
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        return code.strip()

    def _generate_stub_code(self, need: CapabilityNeed) -> str:
        """Generate stub code if LLM fails."""
        func = need.suggested_interface.functions[0] if need.suggested_interface.functions else None

        if not func:
            func = FunctionSignature(
                name="execute",
                description=need.description,
                parameters={},
                returns={"type": "Any"},
            )

        params = ", ".join([f"{k}: {v}" for k, v in func.parameters.items()]) if func.parameters else ""
        return f'''"""{need.description}"""

async def {func.name}({params}) -> {func.returns.get("type", "Any")}:
    """
    {func.description}
    """
    # TODO: Implement this capability
    raise NotImplementedError("This capability needs implementation")
'''

    async def _generate_tests(
        self,
        need: CapabilityNeed,
        code: str,
        research: ResearchResult,
    ) -> str:
        """Generate test code."""
        prompt = f"""Generate pytest tests for this capability.

Capability code:
```python
{code}
```

Testing strategy: {research.testing_strategy}

Generate:
1. Import statements
2. Test fixtures if needed
3. Unit tests for the main function
4. Tests for error cases
5. At least 3 test cases

Output ONLY the Python test code, no markdown."""

        try:
            test_code = await self._llm.generate(prompt, temperature=0.2)
            return self._clean_code(test_code)
        except Exception as e:
            logger.warning("Test generation failed, using stub", error=str(e))
            return self._generate_stub_tests(need)

    def _generate_stub_tests(self, need: CapabilityNeed) -> str:
        """Generate stub tests."""
        func_name = need.suggested_interface.functions[0].name if need.suggested_interface.functions else "execute"
        return f'''import pytest

from capability import {func_name}

@pytest.mark.asyncio
async def test_{func_name}_basic():
    """Test basic functionality."""
    result = await {func_name}()
    assert result is not None

@pytest.mark.asyncio
async def test_{func_name}_error_handling():
    """Test error handling."""
    # TODO: Add error case tests
    pass
'''

    # ═══════════════════════════════════════════════════════════════
    # Phase 4: Validation
    # ═══════════════════════════════════════════════════════════════

    async def validate_capability(self, capability: Capability) -> ValidationResult:
        """
        Validate a capability in the sandbox.

        Args:
            capability: The capability to validate

        Returns:
            Validation result
        """
        logger.info("Validating capability", id=capability.manifest.id)

        capability.status = CapabilityStatus.VALIDATING

        result = await self._sandbox.validate(capability)
        capability.validation_result = result

        if result.passed:
            capability.status = CapabilityStatus.ACTIVE
            logger.info("Capability validation passed", id=capability.manifest.id)
        else:
            capability.status = CapabilityStatus.FAILED
            logger.warning(
                "Capability validation failed",
                id=capability.manifest.id,
                errors=result.errors,
            )

        # Publish event
        await self._events.publish(
            CapabilityValidated(
                capability_id=capability.manifest.id,
                passed=result.passed,
                errors=result.errors,
            )
        )

        return result

    # ═══════════════════════════════════════════════════════════════
    # Phase 5: Registration
    # ═══════════════════════════════════════════════════════════════

    async def register_capability(self, capability: Capability) -> bool:
        """
        Register a validated capability.

        Args:
            capability: The validated capability

        Returns:
            True if registered successfully
        """
        if capability.status != CapabilityStatus.ACTIVE:
            logger.error(
                "Cannot register unvalidated capability",
                id=capability.manifest.id,
                status=capability.status.name,
            )
            return False

        try:
            await self._registry.register(capability)

            # Save to disk
            self._save_capability_to_disk(capability)

            # Publish event
            await self._events.publish(
                CapabilityRegistered(
                    capability_id=capability.manifest.id,
                    name=capability.manifest.name,
                    functions=[f.name for f in capability.manifest.interface.functions],
                )
            )

            logger.info(
                "Capability registered",
                id=capability.manifest.id,
                name=capability.manifest.name,
            )
            return True

        except Exception as e:
            logger.error("Registration failed", id=capability.manifest.id, error=str(e))
            return False

    def _save_capability_to_disk(self, capability: Capability) -> None:
        """Save capability files to workspace."""
        cap_dir = self._generations_dir / capability.manifest.id
        cap_dir.mkdir(parents=True, exist_ok=True)

        # Save manifest
        manifest_path = cap_dir / "manifest.json"
        capability.manifest.save(manifest_path)

        # Save code
        code_path = cap_dir / "__init__.py"
        code_path.write_text(capability.code)

        # Save tests
        test_path = cap_dir / "test_capability.py"
        test_path.write_text(capability.test_code)

        logger.debug("Capability saved to disk", path=str(cap_dir))

    # ═══════════════════════════════════════════════════════════════
    # End-to-End Generation
    # ═══════════════════════════════════════════════════════════════

    async def generate_and_register(
        self,
        goal_description: str,
        context: str | None = None,
    ) -> Capability | None:
        """
        Complete flow: detect need -> generate -> validate -> register.

        Args:
            goal_description: The goal to achieve
            context: Additional context

        Returns:
            Registered capability if successful, None otherwise
        """
        # Phase 1: Detect need
        need = await self.identify_capability_gap(goal_description, context)
        if not need:
            logger.info("No new capability needed")
            return None

        logger.info(
            "Capability need identified",
            name=need.name,
            urgency=need.urgency,
        )

        # Phase 2: Research
        research = await self.research_best_practices(need)
        logger.info("Research completed", libraries=len(research.best_libraries))

        # Phase 3: Generate
        capability = await self.generate_capability(need, research)

        # Phase 4: Validate
        validation = await self.validate_capability(capability)
        if not validation.passed:
            logger.error(
                "Capability validation failed",
                id=capability.manifest.id,
                errors=validation.errors,
            )
            return capability  # Return for debugging/fixing

        # Phase 5: Register
        success = await self.register_capability(capability)
        if not success:
            return capability  # Return for debugging

        logger.info(
            "Capability successfully generated and registered",
            id=capability.manifest.id,
            name=capability.manifest.name,
        )
        return capability

    # ═══════════════════════════════════════════════════════════════
    # Monitoring and Improvement
    # ═══════════════════════════════════════════════════════════════

    async def analyze_and_improve(self) -> list[Capability]:
        """
        Analyze existing capabilities and improve if needed.

        Returns:
            List of improved capabilities
        """
        improved = []

        # Get all capabilities
        capability_ids = self._registry.list_capabilities()

        for cap_id in capability_ids:
            capability = self._registry.get(cap_id)
            if not capability:
                continue

            # Check quality score
            score = capability.calculate_quality_score()
            if score < 70:
                logger.info(
                    "Capability needs improvement",
                    id=cap_id,
                    score=score,
                )

                # Attempt to improve
                new_version = await self._improve_capability(capability)
                if new_version:
                    improved.append(new_version)

        return improved

    async def _improve_capability(self, capability: Capability) -> Capability | None:
        """Create an improved version of a capability."""
        logger.info("Improving capability", id=capability.manifest.id)

        # Analyze issues
        issues = []
        if capability.validation_result and capability.validation_result.errors:
            issues.extend(capability.validation_result.errors)

        if capability.calculate_quality_score() < 70:
            issues.append("Low quality score")

        # Generate improved version
        prompt = f"""Improve this capability based on issues found:

Current code:
```python
{capability.code}
```

Issues to fix:
{chr(10).join(f"- {issue}" for issue in issues)}

Provide improved code that:
1. Fixes all the issues
2. Maintains the same interface
3. Improves error handling
4. Adds better documentation
5. Optimizes performance if possible

Output ONLY the improved Python code."""

        try:
            improved_code = await self._llm.generate(prompt, temperature=0.3)
            improved_code = self._clean_code(improved_code)

            # Create new version
            new_capability = capability.create_version(
                new_code=improved_code,
                new_tests=capability.test_code,  # Keep same tests for compatibility
            )

            # Validate and register
            validation = await self.validate_capability(new_capability)
            if validation.passed:
                await self.register_capability(new_capability)
                return new_capability

        except Exception as e:
            logger.error("Improvement failed", error=str(e))

        return None
