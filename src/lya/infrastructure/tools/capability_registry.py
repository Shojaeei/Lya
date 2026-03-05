"""Capability Registry.

Manages the lifecycle of capabilities including registration,
hot-reloading, and discovery.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable

from lya.domain.models.capability import Capability, CapabilityStatus, CapabilityManifest
from lya.infrastructure.config.logging import get_logger

logger = get_logger(__name__)


class CapabilityRegistry:
    """
    Registry for managing capabilities.

    Supports:
    - Dynamic registration
    - Hot-reloading
    - Function discovery
    - Dependency tracking
    """

    def __init__(self, capabilities_dir: Path | None = None):
        self._capabilities: dict[str, Capability] = {}
        self._functions: dict[str, tuple[str, Callable]] = {}  # name -> (cap_id, func)
        self._capabilities_dir = capabilities_dir or Path.home() / ".lya" / "capabilities"
        self._capabilities_dir.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════
    # Registration
    # ═══════════════════════════════════════════════════════════════

    async def register(self, capability: Capability) -> None:
        """
        Register a capability and make its functions available.

        Args:
            capability: The capability to register
        """
        if capability.status != CapabilityStatus.ACTIVE:
            raise ValueError(
                f"Cannot register capability with status {capability.status.name}"
            )

        # Save to disk
        await self._save_capability(capability)

        # Load module
        module = self._load_capability_module(capability)

        # Extract functions
        for func_sig in capability.manifest.interface.functions:
            func_name = func_sig.name

            if hasattr(module, func_name):
                func = getattr(module, func_name)
                full_name = f"{capability.manifest.id}.{func_name}"

                self._functions[full_name] = (capability.manifest.id, func)
                capability.register_function(func_name, func)

                logger.info(
                    "Function registered",
                    name=full_name,
                    capability=capability.manifest.id,
                )
            else:
                logger.warning(
                    "Function not found in module",
                    name=func_name,
                    capability=capability.manifest.id,
                )

        # Store capability
        self._capabilities[capability.manifest.id] = capability

        logger.info(
            "Capability registered",
            id=capability.manifest.id,
            name=capability.manifest.name,
            functions=len(capability.manifest.interface.functions),
        )

    async def unregister(self, capability_id: str) -> None:
        """
        Unregister a capability and remove its functions.

        Args:
            capability_id: ID of capability to remove
        """
        if capability_id not in self._capabilities:
            logger.warning("Capability not found", id=capability_id)
            return

        capability = self._capabilities[capability_id]

        # Remove functions
        to_remove = [
            name for name, (cap_id, _) in self._functions.items()
            if cap_id == capability_id
        ]
        for name in to_remove:
            del self._functions[name]
            logger.info("Function unregistered", name=name)

        # Remove capability
        del self._capabilities[capability_id]

        logger.info("Capability unregistered", id=capability_id)

    async def reload(self, capability_id: str) -> None:
        """
        Hot-reload a capability.

        Args:
            capability_id: ID of capability to reload
        """
        if capability_id not in self._capabilities:
            raise ValueError(f"Capability not found: {capability_id}")

        capability = self._capabilities[capability_id]

        # Unregister
        await self.unregister(capability_id)

        # Reload module
        cap_dir = self._capabilities_dir / capability_id

        try:
            # Re-read manifest
            manifest = CapabilityManifest.from_file(cap_dir / "manifest.json")

            # Re-read code
            code = (cap_dir / "__init__.py").read_text()
            test_code = (cap_dir / "test_capability.py").read_text()

            # Create new capability instance
            new_cap = Capability(
                manifest=manifest,
                code=code,
                test_code=test_code,
                status=CapabilityStatus.ACTIVE,
            )

            # Re-register
            await self.register(new_cap)

            logger.info("Capability reloaded", id=capability_id)

        except Exception as e:
            logger.error("Reload failed", id=capability_id, error=str(e))
            raise

    # ═══════════════════════════════════════════════════════════════
    # Discovery
    # ═══════════════════════════════════════════════════════════════

    def get(self, capability_id: str) -> Capability | None:
        """Get a capability by ID."""
        return self._capabilities.get(capability_id)

    def get_function(self, name: str) -> Callable | None:
        """
        Get a function by name.

        Args:
            name: Full function name (e.g., "capability_id.function_name")

        Returns:
            Function if found, None otherwise
        """
        if name in self._functions:
            _, func = self._functions[name]
            return func

        # Try partial match
        for full_name, (_, func) in self._functions.items():
            if full_name.endswith(f".{name}") or full_name == name:
                return func

        return None

    def list_capabilities(self) -> list[str]:
        """List all registered capability IDs."""
        return list(self._capabilities.keys())

    def list_functions(self) -> list[str]:
        """List all available function names."""
        return list(self._functions.keys())

    def search_capabilities(self, query: str) -> list[Capability]:
        """
        Search capabilities by name or description.

        Args:
            query: Search query

        Returns:
            Matching capabilities
        """
        query = query.lower()
        results = []

        for cap in self._capabilities.values():
            if (
                query in cap.manifest.name.lower()
                or query in cap.manifest.description.lower()
            ):
                results.append(cap)

        return results

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_capabilities": len(self._capabilities),
            "total_functions": len(self._functions),
            "by_status": {
                status.name: sum(
                    1 for c in self._capabilities.values()
                    if c.status == status
                )
                for status in CapabilityStatus
            },
            "capabilities": [
                {
                    "id": c.manifest.id,
                    "name": c.manifest.name,
                    "version": c.manifest.version,
                    "status": c.status.name,
                    "functions": len(c.manifest.interface.functions),
                    "quality_score": c.calculate_quality_score(),
                }
                for c in self._capabilities.values()
            ],
        }

    # ═══════════════════════════════════════════════════════════════
    # Persistence
    # ═══════════════════════════════════════════════════════════════

    async def load_all(self) -> int:
        """
        Load all capabilities from disk.

        Returns:
            Number of capabilities loaded
        """
        count = 0

        for cap_dir in self._capabilities_dir.iterdir():
            if not cap_dir.is_dir():
                continue

            manifest_path = cap_dir / "manifest.json"
            if not manifest_path.exists():
                continue

            try:
                # Load manifest
                manifest = CapabilityManifest.from_file(manifest_path)

                # Load code
                code_path = cap_dir / "__init__.py"
                test_path = cap_dir / "test_capability.py"

                if not code_path.exists():
                    continue

                code = code_path.read_text()
                test_code = test_path.read_text() if test_path.exists() else ""

                # Create capability
                capability = Capability(
                    manifest=manifest,
                    code=code,
                    test_code=test_code,
                    status=CapabilityStatus.ACTIVE,
                )

                # Register
                await self.register(capability)
                count += 1

            except Exception as e:
                logger.error("Failed to load capability", path=str(cap_dir), error=str(e))

        logger.info("Capabilities loaded", count=count)
        return count

    async def _save_capability(self, capability: Capability) -> None:
        """Save capability to disk."""
        cap_dir = self._capabilities_dir / capability.manifest.id
        cap_dir.mkdir(parents=True, exist_ok=True)

        # Save manifest
        manifest_path = cap_dir / "manifest.json"
        capability.manifest.save(manifest_path)

        # Save code
        code_path = cap_dir / "__init__.py"
        code_path.write_text(capability.code)

        # Save tests
        test_path = cap_dir / "test_capability.py"
        test_path.write_text(capability.test_code)

        logger.debug("Capability saved to disk", path=str(cap_dir))

    # ═══════════════════════════════════════════════════════════════
    # Module Loading
    # ═══════════════════════════════════════════════════════════════

    def _load_capability_module(self, capability: Capability) -> Any:
        """
        Load capability as a Python module.

        Args:
            capability: Capability to load

        Returns:
            Loaded module
        """
        cap_dir = self._capabilities_dir / capability.manifest.id
        module_path = cap_dir / "__init__.py"

        # Create unique module name
        module_name = f"lya_capabilities.{capability.manifest.id}"

        # Remove if already loaded (for reloading)
        if module_name in sys.modules:
            del sys.modules[module_name]

        # Load module
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if not spec or not spec.loader:
            raise ImportError(f"Could not load module from {module_path}")

        module = importlib.util.module_from_spec(spec)

        # Install dependencies first
        self._install_dependencies(capability.manifest.dependencies)

        # Execute module
        spec.loader.exec_module(module)

        return module

    def _install_dependencies(self, dependencies: list[str]) -> None:
        """Install required dependencies."""
        if not dependencies:
            return

        # Check which are already installed
        to_install = []
        for dep in dependencies:
            package_name = dep.split("[")[0].split("==")[0].split(">=")[0].strip()
            try:
                importlib.import_module(package_name.replace("-", "_"))
            except ImportError:
                to_install.append(dep)

        if to_install:
            logger.info("Installing dependencies", packages=to_install)
            try:
                import subprocess
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q"] + to_install,
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as e:
                logger.error("Failed to install dependencies", error=str(e))
                raise


# Global registry instance
_registry: CapabilityRegistry | None = None


def get_registry() -> CapabilityRegistry:
    """Get global capability registry."""
    global _registry
    if _registry is None:
        _registry = CapabilityRegistry()
    return _registry
