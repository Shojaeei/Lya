"""Self-Improvement Service.

Main service for agent self-improvement through code generation.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from lya.domain.models.improvement import (
    Improvement,
    ImprovementStatus,
    ImprovementType,
    CodeChange,
    ImprovementStats,
)
from lya.infrastructure.self_improvement.tool_generator import ToolGenerator
from lya.infrastructure.config.logging import get_logger

logger = get_logger(__name__)


class SelfImprovementService:
    """
    Service for agent self-improvement.

    Allows the agent to:
    - Generate new tools from natural language descriptions
    - Improve existing tools
    - Track improvement history
    - Rollback changes if needed
    """

    def __init__(self, workspace: Path, llm_interface, tool_registry=None):
        self.workspace = workspace
        self.generator = ToolGenerator(workspace, llm_interface)
        self.tool_registry = tool_registry
        self.tools_dir = workspace / "tools" / "generated"
        self.tools_dir.mkdir(parents=True, exist_ok=True)

        # Improvement tracking
        self.improvements_file = workspace / "memory" / "improvements.json"
        self.improvements_file.parent.mkdir(parents=True, exist_ok=True)
        self.improvements: list[Improvement] = []
        self.stats = ImprovementStats()

        # Load history
        self._load_improvements()

    def _load_improvements(self) -> None:
        """Load improvement history."""
        if self.improvements_file.exists():
            try:
                data = json.loads(self.improvements_file.read_text())
                for item in data.get("improvements", []):
                    improvement = Improvement(
                        goal=item["goal"],
                        improvement_type=ImprovementType[item["improvement_type"]],
                        id=item.get("id", str(uuid4())),
                        status=ImprovementStatus[item["status"]],
                        description=item.get("description", ""),
                        test_cases=item.get("test_cases", []),
                        created_at=datetime.fromisoformat(item["created_at"]),
                        completed_at=datetime.fromisoformat(item["completed_at"])
                        if item.get("completed_at")
                        else None,
                        error=item.get("error"),
                        metadata=item.get("metadata", {}),
                    )
                    self.improvements.append(improvement)
                    self.stats.record(improvement)
            except Exception as e:
                logger.error("Failed to load improvements", error=str(e))

    def _save_improvements(self) -> None:
        """Save improvement history."""
        try:
            data = {
                "improvements": [i.to_dict() for i in self.improvements],
                "stats": self.stats.to_dict(),
                "updated": datetime.now(timezone.utc).isoformat(),
            }
            self.improvements_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error("Failed to save improvements", error=str(e))

    async def improve(
        self,
        goal: str,
        improvement_type: ImprovementType = ImprovementType.NEW_TOOL,
        tool_name: str | None = None,
        test_cases: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Self-improve: generate a new capability to achieve a goal.

        Args:
            goal: Natural language description of what to achieve
            improvement_type: Type of improvement
            tool_name: Optional name for generated tool
            test_cases: Optional test cases

        Returns:
            Improvement result
        """
        logger.info("Starting self-improvement", goal=goal, type=improvement_type.name)

        # Create improvement record
        improvement = Improvement(
            goal=goal,
            improvement_type=improvement_type,
            description=f"Generate tool to: {goal}",
            test_cases=test_cases or [],
        )
        self.improvements.append(improvement)

        improvement.mark_generating()

        try:
            # Generate the tool
            result = await self.generator.generate_tool(
                description=goal,
                tool_name=tool_name,
                test_cases=test_cases,
            )

            if not result["success"]:
                improvement.mark_failed(result.get("error", "Generation failed"))
                self._save_improvements()
                return {
                    "success": False,
                    "stage": "generation",
                    "error": result.get("error"),
                    "improvement_id": str(improvement.id),
                }

            # Record the code change
            change = CodeChange(
                file_path=result["tool_path"],
                new_code=result["code"],
                description=f"Generated tool: {result['tool_name']}",
            )
            improvement.add_change(change)
            improvement.mark_validating()

            # Register with tool registry if available
            if self.tool_registry:
                await self._register_tool(result)

            # Mark completed
            improvement.mark_completed()
            self.stats.record(improvement)
            self._save_improvements()

            logger.info(
                "Self-improvement completed",
                tool_name=result["tool_name"],
                improvement_id=str(improvement.id),
            )

            return {
                "success": True,
                "improvement_id": str(improvement.id),
                "tool_name": result["tool_name"],
                "tool_path": result["tool_path"],
                "message": f"Successfully generated and attached tool: {result['tool_name']}",
            }

        except Exception as e:
            logger.error("Self-improvement failed", error=str(e))
            improvement.mark_failed(str(e))
            self.stats.record(improvement)
            self._save_improvements()

            return {
                "success": False,
                "stage": "execution",
                "error": str(e),
                "improvement_id": str(improvement.id),
            }

    async def iterate(
        self,
        tool_name: str,
        improvement_request: str,
    ) -> dict[str, Any]:
        """
        Improve an existing tool based on feedback.

        Args:
            tool_name: Name of tool to improve
            improvement_request: What to improve

        Returns:
            Improvement result
        """
        logger.info(
            "Iterating on tool",
            tool_name=tool_name,
            request=improvement_request[:100],
        )

        # Create improvement record
        improvement = Improvement(
            goal=improvement_request,
            improvement_type=ImprovementType.TOOL_ENHANCEMENT,
            description=f"Improve {tool_name}: {improvement_request}",
        )
        self.improvements.append(improvement)
        improvement.mark_generating()

        try:
            # Get current code
            current_code = self.generator.get_tool_code(tool_name)
            if not current_code:
                improvement.mark_failed(f"Tool {tool_name} not found")
                self._save_improvements()
                return {
                    "success": False,
                    "error": f"Tool {tool_name} not found",
                }

            # Generate improvement
            result = await self.generator.improve_tool(
                tool_name=tool_name,
                improvement_request=improvement_request,
                current_code=current_code,
            )

            if not result["success"]:
                improvement.mark_failed(result.get("error", "Improvement failed"))
                self._save_improvements()
                return {
                    "success": False,
                    "stage": "generation",
                    "error": result.get("error"),
                }

            # Record change
            change = CodeChange(
                file_path=result["tool_path"],
                original_code=current_code,
                new_code=result["code"],
                description=f"Improved {tool_name}: {improvement_request}",
            )
            improvement.add_change(change)

            # Reload tool if registry available
            if self.tool_registry:
                await self._reload_tool(tool_name)

            improvement.mark_completed()
            self.stats.record(improvement)
            self._save_improvements()

            logger.info("Tool iteration completed", tool_name=tool_name)

            return {
                "success": True,
                "tool_name": tool_name,
                "backup_path": result.get("backup_path"),
                "message": f"Successfully improved {tool_name}",
            }

        except Exception as e:
            logger.error("Tool iteration failed", error=str(e))
            improvement.mark_failed(str(e))
            self.stats.record(improvement)
            self._save_improvements()

            return {
                "success": False,
                "error": str(e),
            }

    async def rollback(self, tool_name: str) -> dict[str, Any]:
        """
        Rollback a tool to its previous version.

        Args:
            tool_name: Name of tool to rollback

        Returns:
            Rollback result
        """
        tool_path = self.tools_dir / f"{tool_name}.py"
        backup_path = tool_path.with_suffix(".py.bak")

        if not backup_path.exists():
            return {
                "success": False,
                "error": f"No backup found for {tool_name}",
            }

        try:
            # Restore backup
            shutil.copy2(backup_path, tool_path)

            # Reload tool
            if self.tool_registry:
                await self._reload_tool(tool_name)

            logger.info("Tool rolled back", tool_name=tool_name)

            return {
                "success": True,
                "tool_name": tool_name,
                "message": f"Rolled back {tool_name} to previous version",
            }

        except Exception as e:
            logger.error("Rollback failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
            }

    async def _register_tool(self, generation_result: dict[str, Any]) -> None:
        """Register a generated tool with the registry."""
        if not self.tool_registry:
            return

        # This would require dynamically loading and registering
        # For now, we just log it
        logger.info(
            "Would register tool",
            tool_name=generation_result.get("tool_name"),
        )

    async def _reload_tool(self, tool_name: str) -> None:
        """Reload a tool after modification."""
        if not self.tool_registry:
            return

        logger.info("Would reload tool", tool_name=tool_name)

    def get_improvement_history(
        self,
        limit: int = 100,
        successful_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Get improvement history."""
        improvements = self.improvements

        if successful_only:
            improvements = [i for i in improvements if i.status == ImprovementStatus.COMPLETED]

        return [i.to_dict() for i in improvements[-limit:]]

    def get_stats(self) -> dict[str, Any]:
        """Get improvement statistics."""
        return self.stats.to_dict()

    def list_generated_tools(self) -> list[str]:
        """List all generated tools."""
        return self.generator.list_generated_tools()


from uuid import uuid4
from datetime import datetime, timezone
