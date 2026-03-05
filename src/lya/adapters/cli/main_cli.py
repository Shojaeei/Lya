"""CLI Interface for Lya.

Command-line interface for interacting with the agent.
Pure Python 3.14+ compatible.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from lya.core.agent_core import AgentCore, AgentConfig, create_agent, run_agent_interactive
from lya.infrastructure.config.logging import get_logger

logger = get_logger(__name__)


class LyaCLI:
    """
    Command-line interface for Lya.

    Provides commands for:
    - Starting the agent
    - Sending messages
    - Managing tools
    - Configuration
    - Monitoring
    """

    def __init__(self) -> None:
        """Initialize CLI."""
        self.parser = self._create_parser()
        self.agent: AgentCore | None = None

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser."""
        parser = argparse.ArgumentParser(
            prog="lya",
            description="Lya - Autonomous Agent System",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  lya start                    # Start interactive mode
  lya start --daemon          # Start in background
  lya message "Hello"          # Send a message
  lya tool exec "ls -la"       # Execute command
  lya status                  # Show status
  lya config --show           # Show configuration
  lya health                  # Check health
            """,
        )

        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Start command
        start_parser = subparsers.add_parser("start", help="Start the agent")
        start_parser.add_argument(
            "--daemon", "-d",
            action="store_true",
            help="Run in background",
        )
        start_parser.add_argument(
            "--config", "-c",
            type=str,
            help="Config file path",
        )
        start_parser.add_argument(
            "--name", "-n",
            type=str,
            default="Lya",
            help="Agent name",
        )
        start_parser.add_argument(
            "--port", "-p",
            type=int,
            help="API port",
        )

        # Message command
        msg_parser = subparsers.add_parser("message", help="Send a message")
        msg_parser.add_argument(
            "text",
            type=str,
            help="Message text",
        )
        msg_parser.add_argument(
            "--file", "-f",
            type=str,
            help="Read message from file",
        )
        msg_parser.add_argument(
            "--json", "-j",
            action="store_true",
            help="Output as JSON",
        )

        # Tool command
        tool_parser = subparsers.add_parser("tool", help="Execute tools")
        tool_subparsers = tool_parser.add_subparsers(dest="tool_command")

        # Tool exec
        exec_parser = tool_subparsers.add_parser("exec", help="Execute shell command")
        exec_parser.add_argument("command", type=str, help="Command to execute")

        # Tool list
        tool_subparsers.add_parser("list", help="List available tools")

        # Tool file
        file_parser = tool_subparsers.add_parser("file", help="File operations")
        file_parser.add_argument("operation", choices=["read", "write", "list"])
        file_parser.add_argument("path", type=str)
        file_parser.add_argument("--content", type=str)

        # Status command
        subparsers.add_parser("status", help="Show agent status")

        # Health command
        subparsers.add_parser("health", help="Check health")

        # Config command
        config_parser = subparsers.add_parser("config", help="Configuration")
        config_parser.add_argument(
            "--show",
            action="store_true",
            help="Show current configuration",
        )
        config_parser.add_argument(
            "--set",
            nargs=2,
            metavar=("KEY", "VALUE"),
            help="Set configuration value",
        )
        config_parser.add_argument(
            "--init",
            action="store_true",
            help="Initialize default config",
        )

        # Memory command
        mem_parser = subparsers.add_parser("memory", help="Memory operations")
        mem_subparsers = mem_parser.add_subparsers(dest="mem_command")
        mem_subparsers.add_parser("stats", help="Memory statistics")
        mem_subparsers.add_parser("clear", help="Clear memory")
        mem_subparsers.add_parser("search", help="Search memory")

        # Workflow command
        wf_parser = subparsers.add_parser("workflow", help="Workflow operations")
        wf_subparsers = wf_parser.add_subparsers(dest="wf_command")
        wf_subparsers.add_parser("list", help="List workflows")
        run_wf = wf_subparsers.add_parser("run", help="Run workflow")
        run_wf.add_argument("name", type=str, help="Workflow name")

        # Stop command
        subparsers.add_parser("stop", help="Stop the agent")

        # Logs command
        logs_parser = subparsers.add_parser("logs", help="View logs")
        logs_parser.add_argument(
            "--follow", "-f",
            action="store_true",
            help="Follow log output",
        )
        logs_parser.add_argument(
            "--lines", "-n",
            type=int,
            default=50,
            help="Number of lines",
        )

        return parser

    async def run(self, args: list[str] | None = None) -> int:
        """Run CLI with arguments.

        Args:
            args: Command line arguments

        Returns:
            Exit code
        """
        parsed = self.parser.parse_args(args)

        if not parsed.command:
            self.parser.print_help()
            return 1

        try:
            handler = getattr(self, f"_cmd_{parsed.command}", None)
            if handler:
                return await handler(parsed)
            else:
                print(f"Unknown command: {parsed.command}")
                return 1

        except KeyboardInterrupt:
            print("\nInterrupted")
            return 130
        except Exception as e:
            logger.error("CLI error", error=str(e))
            print(f"Error: {e}")
            return 1

    async def _cmd_start(self, args: argparse.Namespace) -> int:
        """Handle start command."""
        if args.daemon:
            print("Starting Lya in daemon mode...")
            # Would implement proper daemonization
            return 0

        # Load config
        config = AgentConfig(name=args.name)
        if args.port:
            # Would update config
            pass

        # Create and run agent
        self.agent = await create_agent(name=args.name)

        print(f"\n🤖 {args.name} is ready!")
        print("Type 'exit' or press Ctrl+C to quit\n")

        await run_agent_interactive(self.agent)

        return 0

    async def _cmd_message(self, args: argparse.Namespace) -> int:
        """Handle message command."""
        text = args.text

        if args.file:
            text = Path(args.file).read_text()

        if not text:
            print("Error: No message provided")
            return 1

        # Create temporary agent
        agent = await create_agent()

        try:
            result = await agent.process_message(text)

            if args.json:
                print(json.dumps(result, indent=2, default=str))
            else:
                if result.get("success"):
                    print(result.get("response", "No response"))
                else:
                    print(f"Error: {result.get('error', 'Unknown error')}")

        finally:
            await agent.stop()

        return 0

    async def _cmd_tool(self, args: argparse.Namespace) -> int:
        """Handle tool command."""
        agent = await create_agent()

        try:
            if args.tool_command == "exec":
                result = await agent.execute_tool(
                    "execute_command",
                    {"command": args.command}
                )
                print(result.get("stdout", ""))
                if result.get("stderr"):
                    print(f"stderr: {result['stderr']}", file=sys.stderr)

            elif args.tool_command == "list":
                tools = agent.tool_registry.list_tools()
                for tool in tools:
                    print(f"  - {tool.get('name', 'unknown')}: {tool.get('description', '')}")

            elif args.tool_command == "file":
                if args.operation == "read":
                    result = agent.direct_access.read_file(args.path)
                    if result.get("success"):
                        print(result["content"])
                    else:
                        print(f"Error: {result.get('error')}", file=sys.stderr)

                elif args.operation == "write":
                    if not args.content:
                        print("Error: --content required for write", file=sys.stderr)
                        return 1
                    result = agent.direct_access.write_file(args.path, args.content)
                    print("OK" if result.get("success") else f"Error: {result.get('error')}")

                elif args.operation == "list":
                    result = agent.direct_access.list_directory(args.path)
                    for f in result.get("files", []):
                        print(f"  {f['name']}")

        finally:
            await agent.stop()

        return 0

    async def _cmd_status(self, args: argparse.Namespace) -> int:
        """Handle status command."""
        agent = await create_agent()

        try:
            stats = agent.get_stats()

            print(f"Agent: {agent.config.name}")
            print(f"ID: {agent.id}")
            print(f"State: {stats.state.name}")
            print(f"Uptime: {stats.uptime_seconds:.1f}s")
            print(f"Workspace: {agent.workspace}")

        finally:
            await agent.stop()

        return 0

    async def _cmd_health(self, args: argparse.Namespace) -> int:
        """Handle health command."""
        agent = await create_agent()

        try:
            if agent.health_monitor:
                report = await agent.health_monitor.check_health()
                print(f"Status: {report.status.name}")
                print(f"Uptime: {report.uptime_seconds:.1f}s")
                print("\nChecks:")
                for check in report.checks:
                    status_icon = "✓" if check.status.name == "HEALTHY" else "✗"
                    print(f"  {status_icon} {check.name}: {check.message}")
            else:
                print("Health monitoring not enabled")

        finally:
            await agent.stop()

        return 0

    async def _cmd_config(self, args: argparse.Namespace) -> int:
        """Handle config command."""
        from lya.infrastructure.config.settings_v2 import ConfigurationManager

        manager = ConfigurationManager()

        if args.init:
            from lya.infrastructure.config.settings_v2 import AgentSettings
            manager.save(AgentSettings(), "./lya.json")
            print("Configuration initialized at ./lya.json")
            return 0

        if args.show:
            settings = manager.load()
            print(json.dumps(settings.to_dict(), indent=2))
            return 0

        if args.set:
            key, value = args.set
            print(f"Would set {key} = {value}")
            return 0

        return 0

    async def _cmd_memory(self, args: argparse.Namespace) -> int:
        """Handle memory command."""
        agent = await create_agent()

        try:
            if args.mem_command == "stats":
                summary = agent.working_memory.get_summary()
                print(f"Total items: {summary.total_items}")
                print(f"Expired: {summary.expired_items}")
                print(f"Total importance: {summary.total_importance:.2f}")

            elif args.mem_command == "clear":
                agent.working_memory.clear()
                print("Memory cleared")

            elif args.mem_command == "search":
                print("Search requires query parameter")

        finally:
            await agent.stop()

        return 0

    async def _cmd_workflow(self, args: argparse.Namespace) -> int:
        """Handle workflow command."""
        agent = await create_agent()

        try:
            if args.wf_command == "list":
                workflows = agent.workflow_manager.list_workflows()
                for wf in workflows:
                    print(f"  - {wf}")

            elif args.wf_command == "run":
                result = await agent._execute_workflow(args.name, {})
                print(json.dumps(result, indent=2))

        finally:
            await agent.stop()

        return 0

    async def _cmd_stop(self, args: argparse.Namespace) -> int:
        """Handle stop command."""
        print("Stopping Lya...")
        # Would signal daemon to stop
        return 0

    async def _cmd_logs(self, args: argparse.Namespace) -> int:
        """Handle logs command."""
        print(f"Showing last {args.lines} lines...")
        # Would read log file
        return 0


def main() -> None:
    """CLI entry point."""
    cli = LyaCLI()
    try:
        exit_code = asyncio.run(cli.run())
        sys.exit(exit_code)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
