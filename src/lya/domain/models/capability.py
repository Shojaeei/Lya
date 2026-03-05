"""Capability domain model.

A Capability is a self-contained, testable module that Lya can generate
and use to extend its own functionality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable
from uuid import UUID, uuid4
import json


class CapabilityStatus(Enum):
    """Lifecycle status of a capability."""
    DRAFT = auto()
    VALIDATING = auto()
    ACTIVE = auto()
    DEPRECATED = auto()
    FAILED = auto()


class ValidationStatus(Enum):
    """Validation result status."""
    PENDING = auto()
    RUNNING = auto()
    PASSED = auto()
    FAILED = auto()
    TIMEOUT = auto()


@dataclass
class FunctionSignature:
    """Signature of a capability function."""
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    returns: dict[str, Any] = field(default_factory=dict)
    async_: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "returns": self.returns,
            "async": self.async_,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FunctionSignature:
        return cls(
            name=data["name"],
            description=data["description"],
            parameters=data.get("parameters", {}),
            returns=data.get("returns", {}),
            async_=data.get("async", True),
        )


@dataclass
class CapabilityInterface:
    """Interface definition for a capability."""
    functions: list[FunctionSignature] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "functions": [f.to_dict() for f in self.functions],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CapabilityInterface:
        return cls(
            functions=[FunctionSignature.from_dict(f) for f in data.get("functions", [])],
        )


@dataclass
class SafetyConfig:
    """Safety configuration for capability execution."""
    sandbox: bool = True
    network_access: bool = False
    file_system_access: bool = False
    allowed_domains: list[str] = field(default_factory=list)
    timeout_seconds: int = 30
    max_memory_mb: int = 128
    cpu_limit_percent: int = 50

    def to_dict(self) -> dict[str, Any]:
        return {
            "sandbox": self.sandbox,
            "network_access": self.network_access,
            "file_system_access": self.file_system_access,
            "allowed_domains": self.allowed_domains,
            "timeout_seconds": self.timeout_seconds,
            "max_memory_mb": self.max_memory_mb,
            "cpu_limit_percent": self.cpu_limit_percent,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SafetyConfig:
        return cls(
            sandbox=data.get("sandbox", True),
            network_access=data.get("network_access", False),
            file_system_access=data.get("file_system_access", False),
            allowed_domains=data.get("allowed_domains", []),
            timeout_seconds=data.get("timeout_seconds", 30),
            max_memory_mb=data.get("max_memory_mb", 128),
            cpu_limit_percent=data.get("cpu_limit_percent", 50),
        )


@dataclass
class TestCase:
    """Test case for a capability."""
    name: str
    description: str
    input_data: dict[str, Any] = field(default_factory=dict)
    expected_output: Any = None
    assertions: list[str] = field(default_factory=list)
    should_succeed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input": self.input_data,
            "expected_output": self.expected_output,
            "assertions": self.assertions,
            "should_succeed": self.should_succeed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TestCase:
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            input_data=data.get("input", {}),
            expected_output=data.get("expected_output"),
            assertions=data.get("assertions", []),
            should_succeed=data.get("should_succeed", True),
        )


@dataclass
class ValidationResult:
    """Result of validating a capability."""
    status: ValidationStatus
    exit_code: int
    output: str
    test_results: list[dict[str, Any]] = field(default_factory=list)
    coverage_percent: float = 0.0
    execution_time_ms: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == ValidationStatus.PASSED

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.name,
            "exit_code": self.exit_code,
            "output": self.output,
            "test_results": self.test_results,
            "coverage_percent": self.coverage_percent,
            "execution_time_ms": self.execution_time_ms,
            "errors": self.errors,
            "warnings": self.warnings,
        }


@dataclass
class CapabilityManifest:
    """Manifest describing a capability."""
    id: str
    name: str
    version: str
    description: str
    author: str
    created_at: datetime
    updated_at: datetime
    dependencies: list[str] = field(default_factory=list)
    interface: CapabilityInterface = field(default_factory=CapabilityInterface)
    tests: list[TestCase] = field(default_factory=list)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    parent_capability_id: str | None = None  # For versioning
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "dependencies": self.dependencies,
            "interface": self.interface.to_dict(),
            "tests": [t.to_dict() for t in self.tests],
            "safety": self.safety.to_dict(),
            "parent_capability_id": self.parent_capability_id,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CapabilityManifest:
        return cls(
            id=data["id"],
            name=data["name"],
            version=data["version"],
            description=data["description"],
            author=data["author"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            dependencies=data.get("dependencies", []),
            interface=CapabilityInterface.from_dict(data.get("interface", {})),
            tests=[TestCase.from_dict(t) for t in data.get("tests", [])],
            safety=SafetyConfig.from_dict(data.get("safety", {})),
            parent_capability_id=data.get("parent_capability_id"),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_file(cls, path: Path) -> CapabilityManifest:
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())


class Capability:
    """
    A generated capability that extends Lya's functionality.

    This is the runtime representation of a capability,
    containing both the manifest and the actual code.
    """

    def __init__(
        self,
        manifest: CapabilityManifest,
        code: str,
        test_code: str,
        status: CapabilityStatus = CapabilityStatus.DRAFT,
    ):
        self._manifest = manifest
        self._code = code
        self._test_code = test_code
        self._status = status
        self._validation_result: ValidationResult | None = None
        self._compiled_module: Any = None
        self._functions: dict[str, Callable] = {}

    @property
    def manifest(self) -> CapabilityManifest:
        return self._manifest

    @property
    def code(self) -> str:
        return self._code

    @property
    def test_code(self) -> str:
        return self._test_code

    @property
    def status(self) -> CapabilityStatus:
        return self._status

    @status.setter
    def status(self, value: CapabilityStatus) -> None:
        self._status = value

    @property
    def validation_result(self) -> ValidationResult | None:
        return self._validation_result

    @validation_result.setter
    def validation_result(self, result: ValidationResult) -> None:
        self._validation_result = result

    @property
    def is_active(self) -> bool:
        return self._status == CapabilityStatus.ACTIVE

    @property
    def functions(self) -> dict[str, Callable]:
        return self._functions.copy()

    def get_function(self, name: str) -> Callable | None:
        """Get a function by name."""
        return self._functions.get(name)

    def register_function(self, name: str, func: Callable) -> None:
        """Register a compiled function."""
        self._functions[name] = func

    def create_version(self, new_code: str, new_tests: str | None = None) -> Capability:
        """Create a new version of this capability."""
        # Parse version
        major, minor, patch = self._manifest.version.split(".")
        new_version = f"{major}.{minor}.{int(patch) + 1}"

        new_manifest = CapabilityManifest(
            id=f"{self._manifest.id.split('_v')[0]}_{uuid4().hex[:8]}",
            name=self._manifest.name,
            version=new_version,
            description=self._manifest.description,
            author="lya_self",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            dependencies=self._manifest.dependencies.copy(),
            interface=self._manifest.interface,
            tests=self._manifest.tests.copy(),
            safety=self._manifest.safety,
            parent_capability_id=self._manifest.id,
            metadata=self._manifest.metadata.copy(),
        )

        return Capability(
            manifest=new_manifest,
            code=new_code,
            test_code=new_tests or self._test_code,
            status=CapabilityStatus.DRAFT,
        )

    def calculate_quality_score(self) -> float:
        """
        Calculate a quality score based on various factors.

        Returns score 0-100
        """
        score = 0.0

        # Has tests
        if self._test_code and len(self._test_code) > 100:
            score += 20

        # Has validation
        if self._validation_result:
            if self._validation_result.passed:
                score += 30
            # Coverage
            score += min(self._validation_result.coverage_percent * 0.3, 30)

        # Code length (reasonable)
        lines = len(self._code.splitlines())
        if 10 <= lines <= 500:
            score += 10

        # Has documentation
        if '"""' in self._code or "'''" in self._code:
            score += 10

        return min(score, 100)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest": self._manifest.to_dict(),
            "code": self._code,
            "test_code": self._test_code,
            "status": self._status.name,
            "validation": self._validation_result.to_dict() if self._validation_result else None,
            "quality_score": self.calculate_quality_score(),
        }
