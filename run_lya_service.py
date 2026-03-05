"""Lya Background Service Runner.

Runs the Telegram bot as a persistent background process with:
- PID file for management
- Log file output
- Auto-restart on crash (max 5 restarts)
- Cross-platform (Windows/Linux/Mac)

Usage:
    python run_lya_service.py start    # start in background
    python run_lya_service.py stop     # stop the running bot
    python run_lya_service.py status   # check if running
    python run_lya_service.py restart  # restart the bot
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

WORKSPACE = Path(os.environ.get(
    "LYA_WORKSPACE_PATH",
    Path(__file__).parent / "workspace",
))
PID_FILE = WORKSPACE / "lya.pid"
LOG_FILE = WORKSPACE / "lya.log"
SCRIPT = Path(__file__).parent / "run_lya.py"


def _read_pid() -> int | None:
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            return pid
        except (ValueError, OSError):
            return None
    return None


def _is_running(pid: int) -> bool:
    try:
        if sys.platform == "win32":
            # Windows: use tasklist to check
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=5,
            )
            return str(pid) in result.stdout
        else:
            os.kill(pid, 0)
            return True
    except (OSError, subprocess.TimeoutExpired):
        return False


def start():
    """Start bot in background."""
    WORKSPACE.mkdir(parents=True, exist_ok=True)

    existing_pid = _read_pid()
    if existing_pid and _is_running(existing_pid):
        print(f"Bot is already running (PID: {existing_pid})")
        return

    python = sys.executable

    # Use a wrapper script that auto-restarts on crash
    wrapper = WORKSPACE / "_lya_wrapper.py"
    wrapper.write_text(f'''
import subprocess, sys, time
max_restarts = 5
script = r"{SCRIPT}"
for attempt in range(max_restarts):
    print(f"[Service] Starting Lya (attempt {{attempt + 1}}/{{max_restarts}})", flush=True)
    proc = subprocess.run(
        [sys.executable, script],
        env={{**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}},
        cwd=r"{SCRIPT.parent}",
    )
    if proc.returncode == 0:
        break
    print(f"[Service] Lya exited with code {{proc.returncode}}, restarting in 5s...", flush=True)
    time.sleep(5)
print("[Service] Lya stopped.", flush=True)
''', encoding="utf-8")

    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

    if sys.platform == "win32":
        flags = subprocess.CREATE_NO_WINDOW
        log = open(LOG_FILE, "w", encoding="utf-8")
        proc = subprocess.Popen(
            [python, str(wrapper)],
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=str(SCRIPT.parent),
            creationflags=flags,
            env=env,
        )
    else:
        log = open(LOG_FILE, "w", encoding="utf-8")
        proc = subprocess.Popen(
            [python, str(wrapper)],
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=str(SCRIPT.parent),
            start_new_session=True,
            env=env,
        )

    PID_FILE.write_text(str(proc.pid))
    print(f"Bot started (PID: {proc.pid})")
    print(f"Logs: {LOG_FILE}")
    print(f"PID file: {PID_FILE}")


def stop():
    """Stop the running bot."""
    pid = _read_pid()
    if not pid:
        print("No PID file found. Bot may not be running.")
        return

    if not _is_running(pid):
        print(f"Process {pid} is not running. Cleaning up PID file.")
        PID_FILE.unlink(missing_ok=True)
        return

    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/PID", str(pid)],
                capture_output=True, timeout=10,
            )
        else:
            os.kill(pid, signal.SIGTERM)
            # Wait for graceful shutdown
            for _ in range(10):
                time.sleep(0.5)
                if not _is_running(pid):
                    break
            else:
                os.kill(pid, signal.SIGKILL)

        PID_FILE.unlink(missing_ok=True)
        print(f"Bot stopped (PID: {pid})")
    except Exception as e:
        print(f"Error stopping bot: {e}")


def status():
    """Check bot status."""
    pid = _read_pid()
    if not pid:
        print("Bot is NOT running (no PID file)")
        return

    if _is_running(pid):
        print(f"Bot is RUNNING (PID: {pid})")
        print(f"Logs: {LOG_FILE}")
        # Show last 5 lines of log
        if LOG_FILE.exists():
            lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
            if lines:
                print("\nRecent logs:")
                for line in lines[-5:]:
                    print(f"  {line}")
    else:
        print(f"Bot is NOT running (stale PID: {pid})")
        PID_FILE.unlink(missing_ok=True)


def restart():
    """Restart the bot."""
    stop()
    time.sleep(2)
    start()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1].lower()
    commands = {
        "start": start,
        "stop": stop,
        "status": status,
        "restart": restart,
    }

    handler = commands.get(cmd)
    if handler:
        handler()
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python run_lya_service.py [start|stop|status|restart]")


if __name__ == "__main__":
    main()
