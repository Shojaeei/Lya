"""System Tools for Lya.

Shell command execution and system operations.
"""

from __future__ import annotations

import asyncio
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

from lya.infrastructure.config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CommandResult:
    """Result of a command execution."""

    success: bool
    returncode: int
    stdout: str
    stderr: str
    command: str
    error: str | None = None
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "command": self.command,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


class SystemTools:
    """
    System tools for shell operations and system information.

    Features:
    - Shell command execution with safety controls
    - Command existence checking
    - System information gathering
    - Resource monitoring
    """

    # Commands that are blocked for safety
    BLOCKED_COMMANDS = [
        "rm -rf /",
        "rm -rf /*",
        "mkfs",
        "dd if=/dev/zero",
        ":(){ :|: & };:",  # Fork bomb
        "chmod -R 777 /",
        "chown -R",
        "su -",
        "sudo rm",
        "sudo mkfs",
        "format c:",
        "del /f /s /q c:\\",
    ]

    # Partial patterns that trigger blocking
    BLOCKED_PATTERNS = [
        "> /dev/sda",
        "> /dev/hda",
        "> /dev/nvme",
    ]

    def __init__(self, default_timeout: int = 30, shell: bool = True):
        self.default_timeout = default_timeout
        self.shell = shell

    def _check_safety(self, command: str) -> tuple[bool, str | None]:
        """
        Check if command is safe to execute.

        Returns:
            Tuple of (is_safe, error_message)
        """
        cmd_lower = command.lower().strip()

        # Check exact blocked commands
        for blocked in self.BLOCKED_COMMANDS:
            if blocked.lower() in cmd_lower:
                return False, f"Command blocked for safety: {blocked}"

        # Check blocked patterns
        for pattern in self.BLOCKED_PATTERNS:
            if pattern.lower() in cmd_lower:
                return False, f"Command pattern blocked: {pattern}"

        return True, None

    async def execute(
        self,
        command: str,
        cwd: str | Path | None = None,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
        shell: bool | None = None,
    ) -> CommandResult:
        """
        Execute a shell command.

        Args:
            command: Command to execute
            cwd: Working directory
            timeout: Timeout in seconds
            env: Environment variables
            shell: Use shell execution

        Returns:
            CommandResult with output and status
        """
        import time

        start_time = time.time()

        # Safety check
        is_safe, error = self._check_safety(command)
        if not is_safe:
            logger.warning("Blocked unsafe command", command=command, reason=error)
            return CommandResult(
                success=False,
                returncode=-1,
                stdout="",
                stderr="",
                command=command,
                error=error,
            )

        try:
            logger.debug(
                "Executing command",
                command=command[:100],
                cwd=str(cwd) if cwd else None,
            )

            # Run command
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )

            # Wait with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout or self.default_timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                duration_ms = (time.time() - start_time) * 1000
                logger.warning("Command timed out", command=command[:100])
                return CommandResult(
                    success=False,
                    returncode=-1,
                    stdout="",
                    stderr="",
                    command=command,
                    error=f"Command timed out after {timeout or self.default_timeout}s",
                    duration_ms=duration_ms,
                )

            duration_ms = (time.time() - start_time) * 1000

            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")

            success = proc.returncode == 0

            if success:
                logger.debug(
                    "Command succeeded",
                    command=command[:100],
                    duration_ms=duration_ms,
                )
            else:
                logger.warning(
                    "Command failed",
                    command=command[:100],
                    returncode=proc.returncode,
                    duration_ms=duration_ms,
                )

            return CommandResult(
                success=success,
                returncode=proc.returncode,
                stdout=stdout_str,
                stderr=stderr_str,
                command=command,
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error("Command execution failed", command=command[:100], error=str(e))
            return CommandResult(
                success=False,
                returncode=-1,
                stdout="",
                stderr="",
                command=command,
                error=str(e),
                duration_ms=duration_ms,
            )

    def check_command_exists(self, command: str) -> bool:
        """
        Check if a command exists in PATH.

        Args:
            command: Command name to check

        Returns:
            True if command exists
        """
        return shutil.which(command) is not None

    async def which(self, command: str) -> str | None:
        """
        Find the path to a command.

        Args:
            command: Command to find

        Returns:
            Full path to command or None
        """
        result = await self.execute(f"which {command}" if platform.system() != "Windows" else f"where {command}")
        if result.success:
            return result.stdout.strip().split("\n")[0]
        return None

    def get_system_info(self) -> dict[str, Any]:
        """
        Get basic system information.

        Returns:
            System information dictionary
        """
        try:
            return {
                "success": True,
                "platform": platform.system(),
                "platform_version": platform.version(),
                "platform_release": platform.release(),
                "python_version": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "architecture": platform.architecture(),
                "node_name": platform.node(),
            }
        except Exception as e:
            logger.error("Failed to get system info", error=str(e))
            return {
                "success": False,
                "error": str(e),
            }

    def get_resource_usage(self) -> dict[str, Any]:
        """
        Get system resource usage.

        Returns:
            Resource usage information
        """
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            cpu_percent = psutil.cpu_percent(interval=0.1)

            return {
                "success": True,
                "cpu": {
                    "percent": cpu_percent,
                    "count": psutil.cpu_count(),
                    "count_logical": psutil.cpu_count(logical=True),
                },
                "memory": {
                    "total_gb": memory.total / (1024**3),
                    "available_gb": memory.available / (1024**3),
                    "used_gb": memory.used / (1024**3),
                    "percent": memory.percent,
                },
                "disk": {
                    "total_gb": disk.total / (1024**3),
                    "used_gb": disk.used / (1024**3),
                    "free_gb": disk.free / (1024**3),
                    "percent": disk.percent,
                },
            }
        except Exception as e:
            logger.error("Failed to get resource usage", error=str(e))
            return {
                "success": False,
                "error": str(e),
            }

    async def get_process_info(self) -> dict[str, Any]:
        """
        Get information about running processes.

        Returns:
            Process information
        """
        try:
            processes = []
            for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            # Sort by memory usage
            processes.sort(key=lambda x: x.get("memory_percent", 0), reverse=True)

            return {
                "success": True,
                "process_count": len(processes),
                "top_memory": processes[:10],
                "top_cpu": sorted(processes, key=lambda x: x.get("cpu_percent", 0), reverse=True)[:10],
            }
        except Exception as e:
            logger.error("Failed to get process info", error=str(e))
            return {
                "success": False,
                "error": str(e),
            }

    async def ping(self, host: str, count: int = 3) -> dict[str, Any]:
        """
        Ping a host.

        Args:
            host: Host to ping
            count: Number of packets

        Returns:
            Ping result
        """
        system = platform.system().lower()

        if system == "windows":
            cmd = f"ping -n {count} {host}"
        else:
            cmd = f"ping -c {count} {host}"

        result = await self.execute(cmd, timeout=30)

        return {
            "success": result.success,
            "host": host,
            "output": result.stdout,
            "error": result.stderr if not result.success else None,
        }


# Global instance
_system_tools: SystemTools | None = None


def get_system_tools() -> SystemTools:
    """Get global system tools instance."""
    global _system_tools
    if _system_tools is None:
        _system_tools = SystemTools()
    return _system_tools
