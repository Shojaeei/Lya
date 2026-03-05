#!/usr/bin/env python3
"""
Lya Unified Installer & Setup
==============================
Single command to setup environment, install dependencies, and configure everything.

Usage:
    python install.py              # Full interactive setup
    python install.py --quick    # Quick setup with defaults
    python install.py --check    # Check installation only
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path
from typing import NoReturn


class Colors:
    """Terminal colors."""
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    END = "\033[0m"
    BOLD = "\033[1m"


def print_banner():
    """Print installation banner."""
    print(f"""
{Colors.CYAN}
╭──────────────────────────────────────────────────────────────╮
│                                                              │
│   🤖 LYA - Autonomous AGI Agent - Setup                      │
│                                                              │
│   Professional • Production-Ready • Clean Architecture       │
│                                                              │
╰──────────────────────────────────────────────────────────────╯
{Colors.END}
""")


def run_command(cmd: list[str], description: str, cwd: Path | None = None) -> bool:
    """Run a shell command with progress indication."""
    print(f"{Colors.BLUE}→ {description}...{Colors.END}")
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        if result.stdout:
            print(result.stdout)
        print(f"{Colors.GREEN}✓ {description} completed{Colors.END}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{Colors.FAIL}✗ {description} failed{Colors.END}")
        if e.stderr:
            print(f"{Colors.FAIL}{e.stderr}{Colors.END}")
        return False


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"{Colors.GREEN}✓ Python {version.major}.{version.minor}.{version.micro} (compatible){Colors.END}")
        return True
    else:
        print(f"{Colors.FAIL}✗ Python {version.major}.{version.minor} - requires Python 3.10+{Colors.END}")
        return False


def create_virtual_environment(project_root: Path) -> Path:
    """Create and return virtual environment path."""
    venv_path = project_root / ".venv"

    if venv_path.exists():
        print(f"{Colors.WARNING}! Virtual environment already exists at {venv_path}{Colors.END}")
        response = input(f"{Colors.BLUE}Recreate? (y/N): {Colors.END}").lower()
        if response == 'y':
            import shutil
            shutil.rmtree(venv_path)
        else:
            return venv_path

    print(f"{Colors.BLUE}→ Creating virtual environment...{Colors.END}")
    venv.create(venv_path, with_pip=True)
    print(f"{Colors.GREEN}✓ Virtual environment created at {venv_path}{Colors.END}\n")
    return venv_path


def install_dependencies(venv_path: Path, project_root: Path) -> bool:
    """Install project dependencies."""
    pip_path = venv_path / "bin" / "pip" if os.name != 'nt' else venv_path / "Scripts" / "pip.exe"

    # Upgrade pip first
    if not run_command([str(pip_path), "install", "--upgrade", "pip"], "Upgrading pip"):
        return False

    # Install poetry
    if not run_command([str(pip_path), "install", "poetry"], "Installing Poetry"):
        return False

    poetry_path = venv_path / "bin" / "poetry" if os.name != 'nt' else venv_path / "Scripts" / "poetry.exe"

    # Configure poetry
    run_command(
        [str(poetry_path), "config", "virtualenvs.create", "false"],
        "Configuring Poetry"
    )

    # Install dependencies
    if not run_command(
        [str(poetry_path), "install", "--extras", "all"],
        "Installing dependencies (this may take a few minutes)",
        cwd=project_root
    ):
        return False

    return True


def create_env_file(project_root: Path, interactive: bool = True) -> None:
    """Create .env file with configuration."""
    env_path = project_root / ".env"
    env_example = project_root / ".env.example"

    if env_path.exists():
        print(f"{Colors.WARNING}! .env file already exists{Colors.END}")
        return

    if interactive:
        print(f"\n{Colors.CYAN}=== Configuration ==={Colors.END}\n")

        config = {}

        # LLM Provider
        print(f"{Colors.BOLD}LLM Configuration:{Colors.END}")
        print("1. Ollama (local, recommended for development)")
        print("2. OpenAI")
        print("3. Anthropic")
        choice = input(f"{Colors.BLUE}Select LLM provider [1]: {Colors.END}") or "1"

        if choice == "1":
            config["LYA_LLM_PROVIDER"] = "ollama"
            config["LYA_LLM_MODEL"] = "kimi-k2.5:cloud"
            config["LYA_LLM_BASE_URL"] = "http://localhost:11434"
        elif choice == "2":
            config["LYA_LLM_PROVIDER"] = "openai"
            config["LYA_LLM_MODEL"] = "gpt-4"
            api_key = input(f"{Colors.BLUE}OpenAI API Key: {Colors.END}")
            config["LYA_OPENAI_API_KEY"] = api_key
        elif choice == "3":
            config["LYA_LLM_PROVIDER"] = "anthropic"
            config["LYA_LLM_MODEL"] = "claude-3-sonnet"
            api_key = input(f"{Colors.BLUE}Anthropic API Key: {Colors.END}")
            config["LYA_ANTHROPIC_API_KEY"] = api_key

        # Vector DB
        print(f"\n{Colors.BOLD}Vector Database:{Colors.END}")
        print("1. Chroma (local, embedded)")
        print("2. Qdrant (requires server)")
        db_choice = input(f"{Colors.BLUE}Select vector DB [1]: {Colors.END}") or "1"

        if db_choice == "1":
            config["LYA_MEMORY_VECTOR_DB"] = "chroma"
        else:
            config["LYA_MEMORY_VECTOR_DB"] = "qdrant"
            host = input(f"{Colors.BLUE}Qdrant host [localhost]: {Colors.END}") or "localhost"
            config["LYA_QDRANT_HOST"] = host

        # Telegram
        print(f"\n{Colors.BOLD}Telegram Bot (optional):{Colors.END}")
        telegram_token = input(f"{Colors.BLUE}Bot Token (press Enter to skip): {Colors.END}")
        if telegram_token:
            config["LYA_TELEGRAM_BOT_TOKEN"] = telegram_token
            allowed = input(f"{Colors.BLUE}Allowed usernames (comma-separated): {Colors.END}")
            if allowed:
                config["LYA_TELEGRAM_ALLOWED_USERS"] = allowed

        # Write config
        with open(env_path, "w") as f:
            f.write(f"# Lya Configuration\n")
            f.write(f"# Generated by install.py\n\n")
            f.write(f"LYA_ENV=development\n")
            f.write(f"LYA_DEBUG=true\n")
            f.write(f"LYA_LOG_LEVEL=INFO\n\n")

            for key, value in config.items():
                f.write(f"{key}={value}\n")

        print(f"{Colors.GREEN}✓ Configuration saved to .env{Colors.END}\n")
    else:
        # Copy example
        if env_example.exists():
            import shutil
            shutil.copy(env_example, env_path)
            print(f"{Colors.GREEN}✓ Created .env from template{Colors.END}\n")


def setup_workspace(project_root: Path) -> None:
    """Create workspace directories."""
    workspace = Path.home() / ".lya"
    workspace.mkdir(parents=True, exist_ok=True)

    for subdir in ["logs", "memory", "workspace", "cache"]:
        (workspace / subdir).mkdir(exist_ok=True)

    print(f"{Colors.GREEN}✓ Workspace created at {workspace}{Colors.END}\n")


def print_next_steps(venv_path: Path) -> None:
    """Print next steps."""
    activate_script = "source .venv/bin/activate" if os.name != 'nt' else ".venv\\Scripts\\activate"

    print(f"""
{Colors.GREEN}{Colors.BOLD}✓ Setup Complete!{Colors.END}

{Colors.CYAN}Next Steps:{Colors.END}
1. Activate the virtual environment:
   {Colors.BOLD}{activate_script}{Colors.END}

2. Start Lya:
   {Colors.BOLD}python -m lya run{Colors.END}
   or
   {Colors.BOLD}lya run{Colors.END}

{Colors.CYAN}Available Commands:{Colors.END}
• lya run --autonomous    # Run autonomously
• lya chat               # Interactive mode
• lya api                # Start API server
• lya init               # Initialize workspace

{Colors.CYAN}Documentation:{Colors.END}
• README.md              # Quick start guide
• docs/architecture/       # Architecture docs
• Makefile               # Development commands
""")


def main() -> NoReturn:
    """Main installation flow."""
    parser = argparse.ArgumentParser(description="Lya Setup")
    parser.add_argument("--quick", action="store_true", help="Quick setup with defaults")
    parser.add_argument("--check", action="store_true", help="Check installation only")
    args = parser.parse_args()

    print_banner()

    project_root = Path(__file__).parent.resolve()
    os.chdir(project_root)

    if args.check:
        print(f"{Colors.CYAN}=== Checking Installation ==={Colors.END}\n")
        check_python_version()
        sys.exit(0)

    # Pre-checks
    print(f"{Colors.CYAN}=== Pre-checks ==={Colors.END}\n")
    if not check_python_version():
        sys.exit(1)

    # Setup
    print(f"{Colors.CYAN}=== Setup ==={Colors.END}\n")

    venv_path = create_virtual_environment(project_root)

    if not install_dependencies(venv_path, project_root):
        print(f"{Colors.FAIL}✗ Installation failed{Colors.END}")
        sys.exit(1)

    create_env_file(project_root, interactive=not args.quick)
    setup_workspace(project_root)

    print_next_steps(venv_path)

    # Ask to run
    if not args.quick:
        response = input(f"{Colors.BLUE}Start Lya now? (y/N): {Colors.END}").lower()
        if response == 'y':
            print(f"\n{Colors.CYAN}Starting Lya...{Colors.END}\n")
            # Start the main runner
            subprocess.run([sys.executable, str(project_root / "lya.py")])


if __name__ == "__main__":
    main()
