"""Security Hardening Module for Lya.

Provides comprehensive security features including:
- Input sanitization
- Command validation
- Self-destruct mechanism
- Audit logging
- Rate limiting

Pure Python 3.14+ compatible.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any
from collections.abc import Callable


class SecurityLevel(Enum):
    """Security classification levels."""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    MAXIMUM = auto()


@dataclass
class SecurityEvent:
    """Security event log entry."""
    event_type: str
    timestamp: str
    severity: str
    details: dict[str, Any]
    source_ip: str | None = None
    user_id: str | None = None


class InputValidator:
    """
    Validates and sanitizes user input.

    Prevents injection attacks and malicious input.
    """

    # Dangerous patterns
    DANGEROUS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION)\b",
        r"\b(EVAL|EXEC|SYSTEM|IMPORT)\b",
        r"__import__",
        r"__subclasses__",
        r"__builtins__",
        r"os\.system",
        r"subprocess\.",
        r"eval\s*\(",
        r"exec\s*\(",
        r"compile\s*\(",
        r"\$\{",
        r"`.*?`",
    ]

    # Shell injection patterns
    SHELL_PATTERNS = [
        r"[;&|]",
        r"\$\(",
        r"`",
        r"\|\s*sh",
        r"\|\s*bash",
        r">\s*/dev/",
        r"<\s*/dev/",
    ]

    def __init__(self) -> None:
        """Initialize validator."""
        self._compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.DANGEROUS_PATTERNS]
        self._shell_patterns = [re.compile(p, re.IGNORECASE) for p in self.SHELL_PATTERNS]

    def sanitize_string(self, text: str, max_length: int = 10000) -> str:
        """Sanitize a string input.

        Args:
            text: Input text
            max_length: Maximum allowed length

        Returns:
            Sanitized text
        """
        if not isinstance(text, str):
            return ""

        # Trim length
        text = text[:max_length]

        # Remove null bytes
        text = text.replace("\x00", "")

        # Normalize whitespace
        text = " ".join(text.split())

        return text

    def validate_command(self, command: str) -> tuple[bool, str]:
        """Validate a shell command.

        Args:
            command: Command string

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not command or not isinstance(command, str):
            return False, "Empty or invalid command"

        # Check for dangerous shell patterns
        for pattern in self._shell_patterns:
            if pattern.search(command):
                return False, f"Dangerous shell pattern detected"

        return True, ""

    def validate_path(self, path: str, allowed_prefixes: list[str] | None = None) -> tuple[bool, str]:
        """Validate a file path.

        Args:
            path: File path
            allowed_prefixes: List of allowed path prefixes

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            resolved = Path(path).resolve()

            # Check for path traversal
            normalized = str(resolved)
            if ".." in normalized or normalized.startswith("//"):
                return False, "Path traversal detected"

            # Check allowed prefixes
            if allowed_prefixes:
                allowed = any(
                    str(resolved).startswith(Path(p).resolve())
                    for p in allowed_prefixes
                )
                if not allowed:
                    return False, "Path outside allowed directories"

            # Check for dangerous paths
            dangerous = [
                "/etc/passwd",
                "/etc/shadow",
                "/root/.ssh",
                "/etc/ssh",
            ]
            for d in dangerous:
                if d in str(resolved):
                    return False, f"Access to system path denied: {d}"

            return True, ""

        except Exception as e:
            return False, f"Invalid path: {e}"

    def scan_for_injection(self, text: str) -> tuple[bool, list[str]]:
        """Scan for injection attempts.

        Args:
            text: Text to scan

        Returns:
            Tuple of (is_safe, list_of_issues)
        """
        issues = []

        for pattern in self._compiled_patterns:
            matches = pattern.findall(text)
            if matches:
                issues.append(f"Pattern matched: {matches[0][:50]}")

        # Check for excessive encoding
        encoded_count = text.count("%") + text.count("\\x") + text.count("&#")
        if encoded_count > len(text) * 0.3:
            issues.append("Excessive encoding detected")

        return len(issues) == 0, issues


class RateLimiter:
    """
    Rate limiting for operations.

    Prevents abuse and resource exhaustion.
    """

    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: float = 60.0,
    ) -> None:
        """Initialize rate limiter.

        Args:
            max_requests: Max requests per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = {}

    def is_allowed(self, key: str) -> tuple[bool, dict[str, Any]]:
        """Check if request is allowed.

        Args:
            key: Request identifier (IP, user ID, etc.)

        Returns:
            Tuple of (allowed, info_dict)
        """
        now = time.time()

        # Clean old requests
        if key in self._requests:
            self._requests[key] = [
                t for t in self._requests[key]
                if now - t < self.window_seconds
            ]
        else:
            self._requests[key] = []

        # Check limit
        current_count = len(self._requests[key])

        if current_count >= self.max_requests:
            oldest = min(self._requests[key]) if self._requests[key] else now
            retry_after = self.window_seconds - (now - oldest)

            return False, {
                "retry_after": max(0, retry_after),
                "limit": self.max_requests,
                "current": current_count,
            }

        # Record request
        self._requests[key].append(now)

        return True, {
            "limit": self.max_requests,
            "remaining": self.max_requests - current_count - 1,
            "window": self.window_seconds,
        }

    def reset(self, key: str) -> None:
        """Reset rate limit for a key."""
        self._requests.pop(key, None)


class AuditLogger:
    """
    Security audit logging.

    Logs all security-relevant events.
    """

    def __init__(self, log_dir: str | Path | None = None) -> None:
        """Initialize audit logger.

        Args:
            log_dir: Directory for log files
        """
        self.log_dir = Path(log_dir) if log_dir else Path("./audit_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._events: list[SecurityEvent] = []
        self._max_buffer = 100

    def log(
        self,
        event_type: str,
        details: dict[str, Any],
        severity: str = "INFO",
        source_ip: str | None = None,
        user_id: str | None = None,
    ) -> None:
        """Log a security event.

        Args:
            event_type: Type of event
            details: Event details
            severity: Severity level
            source_ip: Source IP address
            user_id: User identifier
        """
        event = SecurityEvent(
            event_type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            severity=severity,
            details=details,
            source_ip=source_ip,
            user_id=user_id,
        )

        self._events.append(event)

        # Write to file
        self._write_event(event)

        # Trim buffer
        if len(self._events) > self._max_buffer:
            self._events.pop(0)

    def _write_event(self, event: SecurityEvent) -> None:
        """Write event to log file."""
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_file = self.log_dir / f"audit_{date}.log"

        entry = json.dumps({
            "type": event.event_type,
            "timestamp": event.timestamp,
            "severity": event.severity,
            "details": event.details,
            "source_ip": event.source_ip,
            "user_id": event.user_id,
        }) + "\n"

        with open(log_file, "a") as f:
            f.write(entry)

    def get_events(
        self,
        event_type: str | None = None,
        severity: str | None = None,
        since: datetime | None = None,
    ) -> list[SecurityEvent]:
        """Get filtered events from buffer."""
        events = self._events

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if severity:
            events = [e for e in events if e.severity == severity]

        if since:
            events = [e for e in events if datetime.fromisoformat(e.timestamp) >= since]

        return events


class SelfDestruct:
    """
    Self-destruct mechanism for sensitive data.

    Wipes data on compromise detection.
    """

    def __init__(
        self,
        workspace: str | Path,
        trigger_file: str | Path | None = None,
    ) -> None:
        """Initialize self-destruct.

        Args:
            workspace: Workspace to protect
            trigger_file: File that triggers destruction
        """
        self.workspace = Path(workspace)
        self.trigger_file = Path(trigger_file) if trigger_file else self.workspace / ".self_destruct"

        self._armed = False
        self._check_interval = 60  # seconds

    def arm(self) -> None:
        """Arm the self-destruct mechanism."""
        self._armed = True
        self.trigger_file.touch()

    def disarm(self) -> None:
        """Disarm the self-destruct mechanism."""
        self._armed = False
        if self.trigger_file.exists():
            self.trigger_file.unlink()

    def check_trigger(self) -> bool:
        """Check if self-destruct should trigger.

        Returns:
            True if trigger conditions met
        """
        if not self._armed:
            return False

        # Check for trigger file removal (tampering)
        if not self.trigger_file.exists():
            return True

        # Check for unauthorized access patterns
        # This would integrate with audit logs

        return False

    def execute(self, confirm: bool = False) -> dict[str, Any]:
        """Execute self-destruct.

        Args:
            confirm: Must be True to execute

        Returns:
            Destruction report
        """
        if not confirm:
            return {"success": False, "error": "Confirmation required"}

        report = {
            "success": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "files_destroyed": 0,
            "directories_destroyed": 0,
            "errors": [],
        }

        # Overwrite and delete files
        for item in self.workspace.rglob("*"):
            try:
                if item.is_file():
                    # Overwrite with random data
                    with open(item, "wb") as f:
                        f.write(os.urandom(1024))

                    # Overwrite with zeros
                    with open(item, "wb") as f:
                        f.write(b"\x00" * 1024)

                    item.unlink()
                    report["files_destroyed"] += 1

                elif item.is_dir():
                    report["directories_destroyed"] += 1

            except Exception as e:
                report["errors"].append(f"{item}: {e}")

        # Remove directories
        for item in sorted(self.workspace.rglob("*"), key=lambda x: len(str(x)), reverse=True):
            try:
                if item.is_dir():
                    item.rmdir()
            except Exception:
                pass

        # Clear shell history
        try:
            history_file = Path.home() / ".bash_history"
            if history_file.exists():
                history_file.write_text("")
        except Exception:
            pass

        return report


class SecurityManager:
    """
    Main security manager.

    Coordinates all security features.
    """

    def __init__(
        self,
        security_level: SecurityLevel = SecurityLevel.HIGH,
        workspace: str | Path | None = None,
    ) -> None:
        """Initialize security manager.

        Args:
            security_level: Security level to enforce
            workspace: Workspace path
        """
        self.level = security_level
        self.workspace = Path(workspace) if workspace else Path(".")

        self.validator = InputValidator()
        self.rate_limiter = RateLimiter()
        self.audit_logger = AuditLogger(self.workspace / "audit_logs")

        self._self_destruct: SelfDestruct | None = None
        if security_level == SecurityLevel.MAXIMUM:
            self._self_destruct = SelfDestruct(self.workspace)

    def validate_input(self, text: str, context: str = "general") -> tuple[bool, str]:
        """Validate user input.

        Args:
            text: Input text
            context: Validation context

        Returns:
            Tuple of (is_valid, error_message)
        """
        sanitized = self.validator.sanitize_string(text)

        is_safe, issues = self.validator.scan_for_injection(sanitized)

        if not is_safe:
            self.audit_logger.log(
                event_type="input_validation_failed",
                details={
                    "context": context,
                    "issues": issues,
                },
                severity="WARNING",
            )
            return False, f"Suspicious input detected: {issues[0]}"

        return True, ""

    def validate_command(self, command: str) -> tuple[bool, str]:
        """Validate shell command."""
        return self.validator.validate_command(command)

    def validate_path(self, path: str) -> tuple[bool, str]:
        """Validate file path."""
        allowed_prefixes = [str(self.workspace), "/tmp", str(Path.home())]
        return self.validator.validate_path(path, allowed_prefixes)

    def check_rate_limit(self, key: str) -> tuple[bool, dict[str, Any]]:
        """Check rate limit."""
        return self.rate_limiter.is_allowed(key)

    def arm_self_destruct(self) -> None:
        """Arm self-destruct mechanism."""
        if self._self_destruct:
            self._self_destruct.arm()
            self.audit_logger.log(
                event_type="self_destruct_armed",
                details={"workspace": str(self.workspace)},
                severity="CRITICAL",
            )

    def check_integrity(self) -> dict[str, Any]:
        """Check system integrity."""
        issues = []

        # Check workspace permissions
        if self.workspace.exists():
            stat = self.workspace.stat()
            if stat.st_mode & 0o077:
                issues.append("Workspace has overly permissive permissions")

        # Check for unauthorized files
        suspicious = [
            ".ssh", "id_rsa", "id_dsa",
            "authorized_keys", ".pgpass",
        ]
        for item in self.workspace.rglob("*"):
            if any(s in item.name for s in suspicious):
                issues.append(f"Suspicious file found: {item}")

        return {
            "success": len(issues) == 0,
            "issues": issues,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def generate_report(self) -> dict[str, Any]:
        """Generate security report."""
        recent_events = self.audit_logger.get_events(
            since=datetime.now(timezone.utc) - timedelta(hours=24)
        )

        return {
            "security_level": self.level.name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "workspace": str(self.workspace),
            "integrity_check": self.check_integrity(),
            "recent_events_count": len(recent_events),
            "event_summary": {
                "critical": sum(1 for e in recent_events if e.severity == "CRITICAL"),
                "warning": sum(1 for e in recent_events if e.severity == "WARNING"),
                "info": sum(1 for e in recent_events if e.severity == "INFO"),
            },
        }


# ═══════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Create security manager
    security = SecurityManager(
        security_level=SecurityLevel.HIGH,
        workspace="./test_workspace",
    )

    # Validate input
    result, error = security.validate_input("Hello world")
    print(f"Input validation: {result} - {error}")

    # Test injection detection
    result, error = security.validate_input("<script>alert('xss')</script>")
    print(f"Injection test: {result} - {error}")

    # Validate command
    result, error = security.validate_command("ls -la")
    print(f"Command validation: {result} - {error}")

    # Validate path
    result, error = security.validate_path("./test.txt")
    print(f"Path validation: {result} - {error}")

    # Check rate limit
    for i in range(5):
        allowed, info = security.check_rate_limit("test_user")
        print(f"Rate limit {i+1}: allowed={allowed}, remaining={info.get('remaining')}")

    # Generate report
    report = security.generate_report()
    print(f"\nSecurity Report:\n{json.dumps(report, indent=2)}")
