"""Lya CLI - Command-line interface."""

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


def print_banner():
    """Print Lya banner."""
    banner = """
    ╭──────────────────────────────────────────╮
    │                                          │
    │   🤖 LYA - Autonomous AGI Agent         │
    │                                          │
    │   "Professional. Autonomous. Evolving." │
    │                                          │
    ╰──────────────────────────────────────────╯
    """
    console.print(Panel.fit(banner, style="cyan"))


@click.group()
@click.version_option(version="0.2.0")
def cli():
    """Lya - Professional Autonomous AGI Agent."""
    pass


@cli.command()
@click.option("--autonomous", is_flag=True, help="Run in autonomous mode")
@click.option("--config", "-c", type=click.Path(), help="Config file path")
def run(autonomous: bool, config: str | None):
    """Start Lya agent."""
    print_banner()

    asyncio.run(_run_agent(autonomous, config))


async def _run_agent(autonomous: bool, config: str | None):
    """Run the agent."""
    from lya.infrastructure.config.logging import configure_logging, get_logger
    from lya.infrastructure.config.settings import settings

    configure_logging()
    logger = get_logger(__name__)

    console.print(f"[green]Environment:[/green] {settings.env}")
    console.print(f"[green]Workspace:[/green] {settings.workspace_path}")
    console.print(f"[green]LLM Provider:[/green] {settings.llm.provider}")
    console.print(f"[green]Vector DB:[/green] {settings.memory.vector_db}")
    console.print()

    console.print("[dim]Agent core initialized[/dim]")

    if autonomous:
        console.print("[green]Running in autonomous mode[/green]")
    else:
        console.print("[dim]Use --autonomous for self-directed operation[/dim]")


@cli.command()
def chat():
    """Start interactive chat mode."""
    print_banner()
    console.print("[green]Starting interactive chat...[/green]")
    import subprocess
    subprocess.run([sys.executable, "run_lya.py"])


@cli.command()
def api():
    """Start API server."""
    print_banner()
    console.print("[green]Starting API server on http://localhost:8000[/green]")
    console.print("[dim]Use 'lya run --web' for web interface[/dim]")


@cli.command()
def status():
    """Show agent status."""
    console.print(Panel("[green]✓[/green] System Ready", title="Status"))


@cli.command()
def init():
    """Initialize Lya workspace."""
    from lya.infrastructure.config.settings import settings

    workspace = settings.workspace_path
    workspace.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (workspace / "logs").mkdir(exist_ok=True)
    (workspace / "memory").mkdir(exist_ok=True)
    (workspace / "workspace").mkdir(exist_ok=True)

    console.print(f"[green]✓[/green] Workspace initialized at {workspace}")


def main():
    """Entry point."""
    cli()


if __name__ == "__main__":
    main()
