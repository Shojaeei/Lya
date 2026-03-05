"""Testing Infrastructure for Lya.

Provides testing utilities and integration tests.
Pure Python 3.14+ compatible.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    duration_ms: float
    error: str | None = None
    output: str = ""


@dataclass
class TestSuiteResult:
    """Result of a test suite."""
    name: str
    tests: list[TestResult]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def passed_count(self) -> int:
        return sum(1 for t in self.tests if t.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for t in self.tests if not t.passed)

    @property
    def total_duration_ms(self) -> float:
        return sum(t.duration_ms for t in self.tests)


class TestCase:
    """Base class for test cases."""

    def setup(self) -> None:
        """Setup before each test."""
        pass

    def teardown(self) -> None:
        """Cleanup after each test."""
        pass

    async def setup_async(self) -> None:
        """Async setup."""
        pass

    async def teardown_async(self) -> None:
        """Async teardown."""
        pass


class LyaTestRunner:
    """Test runner for Lya components."""

    def __init__(self) -> None:
        """Initialize test runner."""
        self._tests: dict[str, Callable] = {}
        self._results: list[TestResult] = []

    def register(self, name: str, test_func: Callable) -> None:
        """Register a test."""
        self._tests[name] = test_func

    def run_test(self, name: str, test_func: Callable) -> TestResult:
        """Run a single test."""
        start = time.time()

        try:
            if asyncio.iscoroutinefunction(test_func):
                result = asyncio.run(test_func())
            else:
                result = test_func()

            duration = (time.time() - start) * 1000

            return TestResult(
                name=name,
                passed=True,
                duration_ms=duration,
                output=str(result) if result else "",
            )

        except Exception as e:
            duration = (time.time() - start) * 1000
            return TestResult(
                name=name,
                passed=False,
                duration_ms=duration,
                error=f"{type(e).__name__}: {e}",
                output=traceback.format_exc(),
            )

    def run_all(self, suite_name: str = "Lya Tests") -> TestSuiteResult:
        """Run all registered tests."""
        results = []

        for name, test_func in self._tests.items():
            print(f"Running {name}...", end=" ")
            result = self.run_test(name, test_func)
            results.append(result)

            status = "PASS" if result.passed else "FAIL"
            print(f"{status} ({result.duration_ms:.1f}ms)")

        return TestSuiteResult(name=suite_name, tests=results)

    def print_report(self, suite_result: TestSuiteResult) -> None:
        """Print test report."""
        print(f"\n{'='*60}")
        print(f"Test Suite: {suite_result.name}")
        print(f"Timestamp: {suite_result.timestamp}")
        print(f"{'='*60}")

        for test in suite_result.tests:
            status = "PASS" if test.passed else "FAIL"
            print(f"\n{status}: {test.name} ({test.duration_ms:.1f}ms)")
            if not test.passed and test.error:
                print(f"  Error: {test.error}")

        print(f"\n{'='*60}")
        print(f"Total: {len(suite_result.tests)} tests")
        print(f"Passed: {suite_result.passed_count}")
        print(f"Failed: {suite_result.failed_count}")
        print(f"Duration: {suite_result.total_duration_ms:.1f}ms")
        print(f"{'='*60}")


class IntegrationTests:
    """Integration tests for Lya components."""

    @staticmethod
    async def test_direct_access() -> bool:
        """Test direct access module."""
        from lya.infrastructure.tools.direct_access import DirectAccess

        da = DirectAccess()

        # Test file operations
        test_file = "/tmp/test_lya.txt"
        da.write_file(test_file, "Hello Lya!")
        result = da.read_file(test_file)

        assert result["success"], "File write failed"
        assert result["content"] == "Hello Lya!", "File content mismatch"

        # Cleanup
        Path(test_file).unlink(missing_ok=True)

        return True

    @staticmethod
    async def test_working_memory() -> bool:
        """Test working memory."""
        from lya.infrastructure.memory.working_buffer import WorkingMemoryBuffer

        wm = WorkingMemoryBuffer(max_items=10)

        # Add items
        id1 = wm.add("Test item 1", importance=0.8, source="test")
        id2 = wm.add("Test item 2", importance=0.5, source="test")

        # Query
        item = wm.get(id1)
        assert item is not None, "Item not found"
        assert item.content == "Test item 1", "Content mismatch"

        return True

    @staticmethod
    async def test_state_graph() -> bool:
        """Test state graph workflows."""
        from lya.infrastructure.workflows.state_graph import StateGraph, State

        graph = StateGraph("test")

        async def step(state: State) -> State:
            new_state = state.copy()
            new_state.set("processed", True)
            return new_state

        graph.add_start_node("start")
        graph.add_node("process", step)
        graph.add_end_node("end")

        graph.add_edge("start", "process")
        graph.add_edge("process", "end")

        result = await graph.execute(State({"input": "test"}))

        assert result.success, f"Workflow failed: {result.error}"
        assert result.final_state.get("processed"), "State not updated"

        return True

    @staticmethod
    async def test_security() -> bool:
        """Test security module."""
        from lya.infrastructure.security.security_hardening import SecurityManager, SecurityLevel

        security = SecurityManager(security_level=SecurityLevel.HIGH)

        # Test input validation
        is_valid, error = security.validate_input("Hello world")
        assert is_valid, f"Valid input rejected: {error}"

        is_valid, error = security.validate_input("<script>alert('xss')</script>")
        assert not is_valid, "Malicious input accepted"

        return True

    @staticmethod
    async def test_health_monitor() -> bool:
        """Test health monitoring."""
        from lya.infrastructure.monitoring.health_monitor import HealthMonitor

        monitor = HealthMonitor(check_interval_seconds=1)
        report = await monitor.check_health()

        assert report is not None, "No health report generated"
        assert len(report.checks) > 0, "No checks performed"

        return True


def run_integration_tests() -> TestSuiteResult:
    """Run all integration tests."""
    runner = LyaTestRunner()

    tests = [
        ("direct_access", IntegrationTests.test_direct_access),
        ("working_memory", IntegrationTests.test_working_memory),
        ("state_graph", IntegrationTests.test_state_graph),
        ("security", IntegrationTests.test_security),
        ("health_monitor", IntegrationTests.test_health_monitor),
    ]

    for name, test_func in tests:
        runner.register(name, test_func)

    result = runner.run_all("Lya Integration Tests")
    runner.print_report(result)

    return result


if __name__ == "__main__":
    print("Running Lya Integration Tests...\n")
    result = run_integration_tests()

    sys.exit(0 if result.failed_count == 0 else 1)
