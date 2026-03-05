"""Testing Utilities for Lya.

Provides test helpers and integration tests for new modules.
Pure Python 3.14+ compatible.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class TestResult:
    """Test execution result."""
    name: str
    success: bool
    duration_ms: float
    error: str | None = None
    details: dict[str, Any] | None = None


class TestRunner:
    """Simple test runner for Lya modules."""

    def __init__(self) -> None:
        """Initialize test runner."""
        self.results: list[TestResult] = []

    async def run_test(
        self,
        name: str,
        test_func,
    ) -> TestResult:
        """Run a single test."""
        start = time.time()

        try:
            await test_func()
            result = TestResult(
                name=name,
                success=True,
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            result = TestResult(
                name=name,
                success=False,
                duration_ms=(time.time() - start) * 1000,
                error=str(e),
            )

        self.results.append(result)
        return result

    def print_report(self) -> None:
        """Print test report."""
        passed = sum(1 for r in self.results if r.success)
        failed = len(self.results) - passed

        print(f"\n{'=' * 50}")
        print(f"Test Results: {passed} passed, {failed} failed")
        print(f"{'=' * 50}")

        for r in self.results:
            status = "✓ PASS" if r.success else "✗ FAIL"
            print(f"{status}: {r.name} ({r.duration_ms:.1f}ms)")
            if r.error:
                print(f"      Error: {r.error}")


class ModuleTests:
    """Tests for Lya modules."""

    @staticmethod
    async def test_direct_access() -> None:
        """Test direct access module."""
        from lya.infrastructure.tools.direct_access import DirectAccess

        direct = DirectAccess()

        # Test file operations
        test_file = "/tmp/lya_test.txt"
        direct.write_file(test_file, "Hello Lya!")
        result = direct.read_file(test_file)
        assert result["success"], f"Read failed: {result}"
        assert result["content"] == "Hello Lya!"

        # Cleanup
        import os
        os.remove(test_file)

    @staticmethod
    async def test_working_memory() -> None:
        """Test working memory."""
        from lya.infrastructure.memory.working_buffer import WorkingMemoryBuffer

        buffer = WorkingMemoryBuffer(max_items=10)

        # Add items
        id1 = buffer.add("Test memory 1", importance=0.8)
        id2 = buffer.add("Test memory 2", importance=0.5)

        # Query
        results = buffer.query("test", top_k=5)
        assert len(results) > 0, "No results found"

        # Get
        item = buffer.get(id1)
        assert item is not None, "Item not found"

    @staticmethod
    async def test_state_graph() -> None:
        """Test state graph workflows."""
        from lya.infrastructure.workflows.state_graph import StateGraph, State

        graph = StateGraph("test")

        async def action(state: State) -> State:
            new_state = state.copy()
            new_state.set("executed", True)
            return new_state

        graph.add_start_node("start")
        graph.add_node("action", action)
        graph.add_end_node("end")
        graph.add_edge("start", "action")
        graph.add_edge("action", "end")

        result = await graph.execute(State())
        assert result.success, f"Workflow failed: {result.error}"
        assert result.final_state.get("executed"), "Action not executed"

    @staticmethod
    async def test_security() -> None:
        """Test security module."""
        from lya.infrastructure.security.security_hardening import SecurityManager

        security = SecurityManager()

        # Test input validation
        is_valid, _ = security.validate_input("Hello world")
        assert is_valid, "Valid input rejected"

        is_valid, _ = security.validate_input("<script>alert('xss')</script>")
        assert not is_valid, "Malicious input accepted"

    @staticmethod
    async def test_health_monitor() -> None:
        """Test health monitoring."""
        from lya.infrastructure.monitoring.health_monitor import HealthMonitor

        monitor = HealthMonitor(check_interval_seconds=1)
        report = await monitor.check_health()

        assert report is not None, "No report generated"
        assert len(report.checks) > 0, "No checks performed"

    @staticmethod
    async def test_capability_generator() -> None:
        """Test capability generator."""
        from lya.infrastructure.self_improvement.capability_generator import (
            CapabilityGenerator, CapabilitySpec
        )

        generator = CapabilityGenerator(output_dir="/tmp/lya_caps")

        spec = CapabilitySpec(
            name="test_capability",
            description="Test capability",
            purpose="Testing",
            parameters=[],
            returns={},
            example_usage="test()",
            test_cases=[],
        )

        capability = generator.generate_capability(spec)
        assert capability.code, "No code generated"
        assert capability.test_code, "No test code generated"


async def run_all_tests() -> None:
    """Run all module tests."""
    runner = TestRunner()

    tests = [
        ("Direct Access", ModuleTests.test_direct_access),
        ("Working Memory", ModuleTests.test_working_memory),
        ("State Graph", ModuleTests.test_state_graph),
        ("Security", ModuleTests.test_security),
        ("Health Monitor", ModuleTests.test_health_monitor),
        ("Capability Generator", ModuleTests.test_capability_generator),
    ]

    for name, test_func in tests:
        print(f"Running: {name}...")
        await runner.run_test(name, test_func)

    runner.print_report()


if __name__ == "__main__":
    asyncio.run(run_all_tests())
