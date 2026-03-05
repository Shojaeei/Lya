"""Security Hardening Module for Lya.

Provides security features including:
- Self-destruct mechanism
- Encrypted memory
- Command validation
- Network isolation
- Audit logging

Pure Python 3.14+ compatible.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class SecurityConfig:
    """Security configuration."""
    enable_self_destruct: bool = True
    enable_encryption: bool = True
    enable_audit_log: bool = True
    enable_command_validation: bool = True
    dangerous_commands: Sequence[str] = field(default_factory=lambda: (
        "rm -rf /", "rm -rf /*", "> /dev/sda", "mkfs", "dd if=/dev/zero",
        "chmod -R 777 /", "chown -R root", "sudo rm", "sudo mkfs",
        "format c:", ":(){ :|: & };:",
    ))
    dangerous_paths: Sequence[str] = field(default_factory=lambda: (
        "/etc/passwd", "/etc/shadow", "/root/.ssh", "/etc/ssh/sshd_config",
    ))
    self_destruct_triggers: Sequence[str] = field(default_factory=lambda: (
        "unauthorized_access", "tamper_detected", "breach_alert",
    ))


@dataclass
class AuditEvent:
    """Security audit event."""
    timestamp: str
    event_type: str
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    source_ip: str = ""
    user: str = ""


class CommandValidator:
    """
    Validates commands for security.

    Blocks dangerous commands and validates paths.
    """

    def __init__(self, config: SecurityConfig | None = None) -> None:
        """Initialize validator.

        Args:
            config: Security configuration
        """
        self.config = config or SecurityConfig()

    def validate_command(self, command: str) -> tuple[bool, str]:
        """Validate a command.

        Args:
            command: Command to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.config.enable_command_validation:
            return True, ""

        cmd_lower = command.lower().strip()

        # Check dangerous commands
        for dangerous in self.config.dangerous_commands:
            if dangerous.lower() in cmd_lower:
                return False, f"Dangerous command blocked: {dangerous}"

        # Check dangerous patterns
        if cmd_lower.startswith("rm -rf /") or cmd_lower.startswith("rm -rf /*"):
            return False, "System-wide deletion blocked"

        if "> /dev/sd" in cmd_lower or "> /dev/hd" in cmd_lower:
            return False, "Device overwrite blocked"

        # Check for command injection patterns
        injection_patterns = [";", "|&", "||", "&&", "`", "$()"]
        suspicious_count = sum(1 for p in injection_patterns if p in cmd_lower)

        if suspicious_count > 2:
            return False, "Suspicious command patterns detected"

        return True, ""

    def validate_path(self, path: str) -> tuple[bool, str]:
        """Validate a file path.

        Args:
            path: Path to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        path_lower = path.lower()

        for dangerous in self.config.dangerous_paths:
            if path_lower.startswith(dangerous.lower()):
                return False, f"Access to {dangerous} blocked"

        return True, ""


class SecureMemory:
    """
    Encrypted in-memory storage.

    Stores sensitive data encrypted in memory.
    """

    def __init__(self, key: bytes | None = None) -> None:
        """Initialize secure memory.

        Args:
            key: Encryption key (generates if None)
        """
        self._key = key or secrets.token_bytes(32)
        self._data: dict[str, bytes] = {}

    def _encrypt(self, data: str) -> bytes:
        """Encrypt data using simple XOR (for demonstration).

        In production, use proper encryption like Fernet or libsodium.
        """
        data_bytes = data.encode()
        key_extended = (self._key * (len(data_bytes) // len(self._key) + 1))[:len(data_bytes)]
        encrypted = bytes(a ^ b for a, b in zip(data_bytes, key_extended))
        return encrypted

    def _decrypt(self, encrypted: bytes) -> str:
        """Decrypt data."""
        key_extended = (self._key * (len(encrypted) // len(self._key) + 1))[:len(encrypted)]
        decrypted = bytes(a ^ b for a, b in zip(encrypted, key_extended))
        return decrypted.decode()

    def store(self, key: str, value: str) -> None:
        """Store encrypted value.

        Args:
            key: Storage key
            value: Value to encrypt and store
        """
        self._data[key] = self._encrypt(value)

    def retrieve(self, key: str) -> str | None:
        """Retrieve and decrypt value.

        Args:
            key: Storage key

        Returns:
            Decrypted value or None
        """
        encrypted = self._data.get(key)
        if encrypted is None:
            return None
        return self._decrypt(encrypted)

    def delete(self, key: str) -> bool:
        """Delete stored value.

        Args:
            key: Storage key

        Returns:
            True if deleted
        """
        if key in self._data:
            # Overwrite before delete
            self._data[key] = b"\x00" * len(self._data[key])
            del self._data[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all data securely."""
        for key in list(self._data.keys()):
            self.delete(key)
        self._data.clear()

    def get_key_hash(self) -> str:
        """Get hash of encryption key (for verification)."""
        return hashlib.sha256(self._key).hexdigest()[:16]


class AuditLogger:
    """
    Security audit logging.

    Logs security events for analysis.
    """

    def __init__(self, log_dir: str | Path | None = None) -> None:
        """Initialize audit logger.

        Args:
            log_dir: Directory for log files
        """
        self.log_dir = Path(log_dir) if log_dir else Path("./security_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._events: list[AuditEvent] = []

    def log(
        self,
        event_type: str,
        severity: str,
        message: str,
        details: dict[str, Any] | None = None,
        source_ip: str = "",
        user: str = "",
    ) -> None:
        """Log an audit event.

        Args:
            event_type: Type of event
            severity: Severity level
            message: Event message
            details: Additional details
            source_ip: Source IP address
            user: User identifier
        """
        event = AuditEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            severity=severity,
            message=message,
            details=details or {},
            source_ip=source_ip,
            user=user,
        )

        self._events.append(event)

        # Write to file
        self._write_event(event)

    def _write_event(self, event: AuditEvent) -> None:
        """Write event to log file."""
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_file = self.log_dir / f"audit_{date_str}.jsonl"

        with open(log_file, "a") as f:
            f.write(json.dumps(event.__dict__) + "\n")

    def get_events(
        self,
        event_type: str | None = None,
        severity: str | None = None,
        since: str | None = None,
    ) -> list[AuditEvent]:
        """Get filtered events.

        Args:
            event_type: Filter by type
            severity: Filter by severity
            since: Filter by timestamp

        Returns:
            Filtered events
        """
        events = self._events

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if severity:
            events = [e for e in events if e.severity == severity]

        if since:
            events = [e for e in events if e.timestamp >= since]

        return events

    def export(self, path: str | Path) -> None:
        """Export all events to file.

        Args:
            path: Export path
        """
        Path(path).write_text(
            json.dumps([e.__dict__ for e in self._events], indent=2)
        )


class SelfDestruct:
    """
    Self-destruct mechanism for Lya.

    Wipes all data when triggered.
    """

    def __init__(
        self,
        workspace: str | Path,
        config: SecurityConfig | None = None,
    ) -> None:
        """Initialize self-destruct.

        Args:
            workspace: Workspace directory to wipe
            config: Security configuration
        """
        self.workspace = Path(workspace)
        self.config = config or SecurityConfig()
        self.trigger_file = self.workspace / ".self_destruct_armed"
        self._armed = False

    def arm(self) -> None:
        """Arm self-destruct mechanism."""
        if not self.config.enable_self_destruct:
            return

        self.trigger_file.touch()
        self._armed = True

    def disarm(self) -> None:
        """Disarm self-destruct mechanism."""
        if self.trigger_file.exists():
            self.trigger_file.unlink()
        self._armed = False

    def is_armed(self) -> bool:
        """Check if self-destruct is armed."""
        return self._armed or self.trigger_file.exists()

    def trigger(self, reason: str = "manual") -> dict[str, Any]:
        """Trigger self-destruct.

        Args:
            reason: Trigger reason

        Returns:
            Destruction report
        """
        if not self.config.enable_self_destruct:
            return {"triggered": False, "reason": "Self-destruct disabled"}

        if not self.is_armed():
            return {"triggered": False, "reason": "Not armed"}

        report = {
            "triggered": True,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "files_wiped": 0,
            "directories_wiped": 0,
        }

        try:
            # Wipe files
            for file_path in self.workspace.rglob("*"):
                if file_path.is_file():
                    # Overwrite with random data
                    try:
                        with open(file_path, "wb") as f:
                            f.write(os.urandom(1024))
                        file_path.unlink()
                        report["files_wiped"] += 1
                    except Exception:
                        pass

                elif file_path.is_dir():
                    report["directories_wiped"] += 1

            # Clear history
            import readline
            readline.clear_history()

            # Clear Python cache
            import gc
            gc.collect()

            # Remove workspace
            import shutil
            try:
                shutil.rmtree(self.workspace, ignore_errors=True)
            except Exception:
                pass

        except Exception as e:
            report["error"] = str(e)

        return report

    def check_triggers(self) -> list[str]:
        """Check for trigger conditions.

        Returns:
            List of trigger conditions detected
        """
        triggers = []

        # Check for unauthorized processes
        try:
            result = subprocess.run(
                ["who"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if len(lines) > 1:  # Multiple users
                    triggers.append("multiple_users")

        except Exception:
            pass

        # Check for network activity (if air-gapped)
        try:
            result = subprocess.run(
                ["ss", "-tuln"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0 and result.stdout.strip():
                # Has network listeners
                triggers.append("network_active")

        except Exception:
            pass

        return triggers


class SecurityManager:
    """
    Main security manager for Lya.

    Coordinates all security features.
    """

    def __init__(
        self,
        workspace: str | Path,
        config: SecurityConfig | None = None,
    ) -> None:
        """Initialize security manager.

        Args:
            workspace: Workspace directory
            config: Security configuration
        """
        self.workspace = Path(workspace)
        self.config = config or SecurityConfig()

        self.validator = CommandValidator(self.config)
        self.secure_memory = SecureMemory()
        self.audit_logger = AuditLogger(self.workspace / "security_logs")
        self.self_destruct = SelfDestruct(self.workspace, self.config)

        # Log initialization
        self.audit_logger.log(
            event_type="security_init",
            severity="info",
            message="Security manager initialized",
        )

    def validate_command(self, command: str) -> tuple[bool, str]:
        """Validate command."""
        is_valid, error = self.validator.validate_command(command)

        if not is_valid:
            self.audit_logger.log(
                event_type="command_blocked",
                severity="warning",
                message=f"Command blocked: {command}",
                details={"error": error},
            )

        return is_valid, error

    def validate_path(self, path: str) -> tuple[bool, str]:
        """Validate path."""
        is_valid, error = self.validator.validate_path(path)

        if not is_valid:
            self.audit_logger.log(
                event_type="path_blocked",
                severity="warning",
                message=f"Path blocked: {path}",
                details={"error": error},
            )

        return is_valid, error

    def store_secret(self, key: str, value: str) -> None:
        """Store secret securely."""
        self.secure_memory.store(key, value)

        self.audit_logger.log(
            event_type="secret_stored",
            severity="info",
            message=f"Secret stored: {key}",
        )

    def get_secret(self, key: str) -> str | None:
        """Get stored secret."""
        value = self.secure_memory.retrieve(key)

        self.audit_logger.log(
            event_type="secret_accessed",
            severity="info",
            message=f"Secret accessed: {key}",
        )

        return value

    def arm_self_destruct(self) -> None:
        """Arm self-destruct."""
        self.self_destruct.arm()

        self.audit_logger.log(
            event_type="self_destruct_armed",
            severity="critical",
            message="Self-destruct armed",
        )

    def disarm_self_destruct(self) -> None:
        """Disarm self-destruct."""
        self.self_destruct.disarm()

        self.audit_logger.log(
            event_type="self_destruct_disarmed",
            severity="info",
            message="Self-destruct disarmed",
        )

    def trigger_self_destruct(self, reason: str) -> dict[str, Any]:
        """Trigger self-destruct."""
        self.audit_logger.log(
            event_type="self_destruct_triggered",
            severity="critical",
            message=f"Self-destruct triggered: {reason}",
        )

        return self.self_destruct.trigger(reason)

    def check_security(self) -> dict[str, Any]:
        """Run security checks."""
        status = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "self_destruct_armed": self.self_destruct.is_armed(),
            "secrets_stored": len(self.secure_memory._data),
            "log_entries": len(self.audit_logger._events),
            "warnings": [],
        }

        # Check for triggers
        triggers = self.self_destruct.check_triggers()
        if triggers:
            status["triggers_detected"] = triggers
            status["warnings"].extend(triggers)

            # Auto-trigger if armed
            if self.self_destruct.is_armed():
                status["auto_triggered"] = True
                status["destruction_report"] = self.trigger_self_destruct(
                    f"Triggers: {', '.join(triggers)}"
                )

        return status

    def get_audit_log(self) -> list[AuditEvent]:
        """Get audit log."""
        return self.audit_logger._events

    def export_security_report(self, path: str | Path) -> None:
        """Export security report."""
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "enable_self_destruct": self.config.enable_self_destruct,
                "enable_encryption": self.config.enable_encryption,
                "enable_audit_log": self.config.enable_audit_log,
                "enable_command_validation": self.config.enable_command_validation,
            },
            "status": self.check_security(),
            "audit_summary": {
                "total_events": len(self.audit_logger._events),
                "event_types": list(set(e.event_type for e in self.audit_logger._events)),
            },
        }

        Path(path).write_text(json.dumps(report, indent=2))


# ═══════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Create security manager
    security = SecurityManager("./lya_workspace")

    # Test command validation
    print("Command validation:")
    commands = [
        "ls -la",
        "rm -rf /",
        "cat /etc/passwd",
    ]

    for cmd in commands:
        valid, error = security.validate_command(cmd)
        print(f"  {cmd}: {'OK' if valid else f'BLOCKED - {error}'}")

    # Test path validation
    print("\nPath validation:")
    paths = [
        "/home/user/file.txt",
        "/etc/passwd",
    ]

    for path in paths:
        valid, error = security.validate_path(path)
        print(f"  {path}: {'OK' if valid else f'BLOCKED - {error}'}")

    # Store and retrieve secret
    print("\nSecure memory:")
    security.store_secret("api_key", "sk-test-12345")
    retrieved = security.get_secret("api_key")
    print(f"  Stored and retrieved: {'*' * len(retrieved) if retrieved else 'None'}")

    # Security check
    print("\nSecurity check:")
    status = security.check_security()
    print(f"  Self-destruct armed: {status['self_destruct_armed']}")
    print(f"  Secrets stored: {status['secrets_stored']}")

    # Export report
    security.export_security_report("./security_report.json")
    print("\nSecurity report exported to security_report.json")
