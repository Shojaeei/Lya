"""Capability Sandbox.

Provides isolated, secure execution environment for testing
generated capabilities before they're integrated into the system.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lya.domain.models.capability import Capability, ValidationResult, ValidationStatus
from lya.infrastructure.config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SandboxConfig:
    """Configuration for sandbox execution."""
    timeout_seconds: int = 60
    max_memory_mb: int = 128
    cpu_limit_percent: int = 50
    network_access: bool = False
    file_system_access: bool = False
    temp_dir: Path | None = None


@dataclass
class SandboxResult:
    """Result of sandbox execution."""
    exit_code: int
    stdout: str
    stderr: str
    execution_time_ms: int
    timed_out: bool = False


class CapabilitySandbox:
    """
    Sandbox for testing generated capabilities.

    Uses RestrictedPython and subprocess isolation to safely
    execute untrusted code.
    """

    # Dangerous patterns to check for
    DANGEROUS_PATTERNS = [
        r"__import__\s*\(",
        r"import\s+os\s*;\s*os\.(system|popen|exec|fork)",
        r"subprocess\.\w+",
        r"eval\s*\(",
        r"exec\s*\(",
        r"compile\s*\(",
        r"__builtins__",
        r"__subclasses__",
        r"\.(system|popen|exec|fork|kill|chmod|chown|remove|unlink|rmdir)",
        r"open\s*\([^)]*['\"]w",
        r"file\s*\(",
        r"socket\.",
        r"urllib\.request",
        r"requests\.(get|post|put|delete|patch)",
        r"wget|curl",
        r"rm\s+-rf",
        r">\s*/dev/null",
        r"\|\s*sh",
        r"\|\s*bash",
    ]

    def __init__(self, config: SandboxConfig | None = None):
        self.config = config or SandboxConfig()
        self._temp_dir = self.config.temp_dir

    def _check_code_safety(self, code: str) -> tuple[bool, list[str]]:
        """
        Static analysis to check for dangerous code patterns.

        Returns:
            Tuple of (is_safe, list_of_issues)
        """
        issues = []

        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                issues.append(f"Potentially dangerous pattern detected: {pattern[:50]}...")

        # Check for very long lines (obfuscation attempt)
        for i, line in enumerate(code.splitlines(), 1):
            if len(line) > 500:
                issues.append(f"Line {i}: Suspiciously long line ({len(line)} chars)")

        # Check for excessive nesting
        max_indent = 0
        for line in code.splitlines():
            if line.strip():
                indent = len(line) - len(line.lstrip())
                max_indent = max(max_indent, indent)
        if max_indent > 40:  # More than 10 levels of nesting
            issues.append(f"Excessive nesting detected ({max_indent // 4} levels)")

        return len(issues) == 0, issues

    async def validate(self, capability: Capability) -> ValidationResult:
        """
        Validate a capability by running its tests in sandbox.

        Args:
            capability: The capability to validate

        Returns:
            ValidationResult with pass/fail status
        """
        logger.info("Starting validation", id=capability.manifest.id)

        # Phase 1: Static analysis
        is_safe, safety_issues = self._check_code_safety(capability.code)
        if not is_safe:
            logger.warning("Safety check failed", issues=safety_issues)
            return ValidationResult(
                status=ValidationStatus.FAILED,
                exit_code=1,
                output="",
                errors=["Static analysis failed"] + safety_issues,
                warnings=[],
            )

        # Phase 2: Syntax check
        syntax_ok, syntax_errors = self._check_syntax(capability.code)
        if not syntax_ok:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                exit_code=1,
                output="",
                errors=syntax_errors,
                warnings=[],
            )

        # Phase 3: Run tests in subprocess
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)

            # Write files
            self._write_capability_files(temp_path, capability)

            # Run tests
            result = await self._run_tests_in_subprocess(temp_path, capability)

        # Parse results
        test_results, coverage = self._parse_test_output(result.stdout, result.stderr)

        # Determine status
        if result.timed_out:
            status = ValidationStatus.TIMEOUT
        elif result.exit_code == 0:
            status = ValidationStatus.PASSED
        else:
            status = ValidationStatus.FAILED

        return ValidationResult(
            status=status,
            exit_code=result.exit_code,
            output=result.stdout + "\n" + result.stderr,
            test_results=test_results,
            coverage_percent=coverage,
            execution_time_ms=result.execution_time_ms,
            errors=[],
            warnings=[],
        )

    def _check_syntax(self, code: str) -> tuple[bool, list[str]]:
        """Check Python syntax."""
        try:
            compile(code, "<string>", "exec")
            return True, []
        except SyntaxError as e:
            return False, [f"Syntax error at line {e.lineno}: {e.msg}"]

    def _write_capability_files(self, temp_path: Path, capability: Capability) -> None:
        """Write capability files to temp directory."""
        # Write main module
        main_file = temp_path / "capability.py"
        main_file.write_text(capability.code)

        # Write test file
        test_file = temp_path / "test_capability.py"
        test_file.write_text(capability.test_code)

        # Write conftest.py
        conftest = temp_path / "conftest.py"
        conftest.write_text("""
import pytest
import asyncio

@pytest.fixture
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
""")

        # Write requirements
        if capability.manifest.dependencies:
            req_file = temp_path / "requirements.txt"
            req_file.write_text("\n".join(capability.manifest.dependencies))

    async def _run_tests_in_subprocess(
        self,
        temp_path: Path,
        capability: Capability,
    ) -> SandboxResult:
        """Run tests in isolated subprocess."""
        cmd = [
            "python", "-m", "pytest",
            "test_capability.py",
            "-v",
            "--tb=short",
            "--timeout=30",
            f"--timeout-method=thread",
        ]

        # Add coverage if pytest-cov available
        try:
            import pytest_cov
            cmd.extend(["--cov=capability", "--cov-report=term-missing"])
        except ImportError:
            pass

        env = {
            "PYTHONPATH": str(temp_path),
            "PYTHONDONTWRITEBYTECODE": "1",
        }

        if not self.config.network_access:
            # This is a hint; true network isolation requires containers
            env["HTTP_PROXY"] = "http://localhost:0"  # Invalid proxy to block
            env["HTTPS_PROXY"] = "http://localhost:0"

        start_time = asyncio.get_event_loop().time()

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.config.timeout_seconds,
                )
                timed_out = False
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                stdout, stderr = b"", b""
                timed_out = True

            execution_time = int(
                (asyncio.get_event_loop().time() - start_time) * 1000
            )

            return SandboxResult(
                exit_code=proc.returncode if proc.returncode else (1 if timed_out else 0),
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                execution_time_ms=execution_time,
                timed_out=timed_out,
            )

        except Exception as e:
            return SandboxResult(
                exit_code=1,
                stdout="",
                stderr=str(e),
                execution_time_ms=int(
                    (asyncio.get_event_loop().time() - start_time) * 1000
                ),
                timed_out=False,
            )

    def _parse_test_output(
        self,
        stdout: str,
        stderr: str,
    ) -> tuple[list[dict[str, Any]], float]:
        """Parse pytest output to extract test results."""
        test_results = []
        coverage = 0.0

        # Parse test lines
        for line in stdout.split("\n"):
            # PASSED
            if "PASSED" in line:
                test_name = line.split("::")[-1].split()[0] if "::" in line else "unknown"
                test_results.append({
                    "name": test_name,
                    "status": "passed",
                    "message": "",
                })
            # FAILED
            elif "FAILED" in line:
                test_name = line.split("::")[-1].split()[0] if "::" in line else "unknown"
                test_results.append({
                    "name": test_name,
                    "status": "failed",
                    "message": line,
                })
            # ERROR
            elif "ERROR" in line:
                test_name = line.split("::")[-1].split()[0] if "::" in line else "unknown"
                test_results.append({
                    "name": test_name,
                    "status": "error",
                    "message": line,
                })

        # Parse coverage
        for line in stdout.split("\n"):
            if "TOTAL" in line and "%" in line:
                try:
                    coverage_str = line.split("%")[0].split()[-1]
                    coverage = float(coverage_str)
                except:
                    pass

        return test_results, coverage


class DockerSandbox(CapabilitySandbox):
    """
    Docker-based sandbox for stronger isolation.

    Requires Docker to be installed and running.
    """

    def __init__(self, config: SandboxConfig | None = None, image: str = "python:3.11-slim"):
        super().__init__(config)
        self.image = image
        self._check_docker()

    def _check_docker(self) -> None:
        """Check if Docker is available."""
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode != 0:
                raise RuntimeError("Docker not running or not accessible")
        except FileNotFoundError:
            raise RuntimeError("Docker not installed")

    async def validate(self, capability: Capability) -> ValidationResult:
        """Validate using Docker container."""
        import tempfile
        import shutil

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)

            # Write files
            self._write_capability_files(temp_path, capability)

            # Create Dockerfile
            dockerfile = temp_path / "Dockerfile"
            dockerfile.write_text(f"""
FROM {self.image}
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt pytest pytest-asyncio pytest-timeout pytest-cov 2>/dev/null || true
COPY . .
CMD ["python", "-m", "pytest", "test_capability.py", "-v", "--tb=short", "--timeout=30"]
""")

            # Build and run
            try:
                # Build image
                build_result = subprocess.run(
                    ["docker", "build", "-t", f"capability-test-{capability.manifest.id}", "."],
                    cwd=temp_path,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )

                if build_result.returncode != 0:
                    return ValidationResult(
                        status=ValidationStatus.FAILED,
                        exit_code=1,
                        output="",
                        errors=[f"Docker build failed: {build_result.stderr}"],
                        warnings=[],
                    )

                # Run container
                run_result = subprocess.run(
                    [
                        "docker", "run",
                        "--rm",
                        "--memory", f"{self.config.max_memory_mb}m",
                        "--cpus", str(self.config.cpu_limit_percent / 100),
                        "--network", "none" if not self.config.network_access else "bridge",
                        "--read-only",
                        f"capability-test-{capability.manifest.id}",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout_seconds,
                )

                # Cleanup
                subprocess.run(
                    ["docker", "rmi", "-f", f"capability-test-{capability.manifest.id}"],
                    capture_output=True,
                )

                # Parse results
                test_results, coverage = self._parse_test_output(
                    run_result.stdout, run_result.stderr
                )

                status = ValidationStatus.PASSED if run_result.returncode == 0 else ValidationStatus.FAILED

                return ValidationResult(
                    status=status,
                    exit_code=run_result.returncode,
                    output=run_result.stdout + "\n" + run_result.stderr,
                    test_results=test_results,
                    coverage_percent=coverage,
                    execution_time_ms=0,  # Would need timing
                    errors=[],
                    warnings=[],
                )

            except subprocess.TimeoutExpired:
                return ValidationResult(
                    status=ValidationStatus.TIMEOUT,
                    exit_code=1,
                    output="",
                    errors=["Docker execution timed out"],
                    warnings=[],
                )
            except Exception as e:
                return ValidationResult(
                    status=ValidationStatus.FAILED,
                    exit_code=1,
                    output="",
                    errors=[f"Docker error: {str(e)}"],
                    warnings=[],
                )
