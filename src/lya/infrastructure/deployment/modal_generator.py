"""Modal Deployment Configuration for Lya.

Provides 24/7 cloud deployment on Modal.com with automatic scaling.
Uses pure Python 3.14+ compatible code.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModalConfig:
    """Configuration for Modal deployment."""

    app_name: str = "lya-agent"
    image_name: str = "lya-image"
    gpu: str | None = None  # "A100", "A10G", "T4", or None
    cpu: float = 2.0
    memory: int = 4096  # MB
    timeout: int = 3600  # seconds
    concurrency_limit: int = 10
    secrets: list[str] = field(default_factory=list)
    mounts: dict[str, str] = field(default_factory=dict)
    pip_packages: list[str] = field(default_factory=list)
    apt_packages: list[str] = field(default_factory=list)
    environment: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "app_name": self.app_name,
            "image_name": self.image_name,
            "gpu": self.gpu,
            "cpu": self.cpu,
            "memory": self.memory,
            "timeout": self.timeout,
            "concurrency_limit": self.concurrency_limit,
            "secrets": self.secrets,
            "mounts": self.mounts,
            "pip_packages": self.pip_packages,
            "apt_packages": self.apt_packages,
            "environment": self.environment,
        }


class ModalDeploymentGenerator:
    """Generates Modal deployment files for Lya."""

    # Default packages needed for Lya
    DEFAULT_PIP_PACKAGES = [
        "fastapi",
        "uvicorn",
        "websockets",
        "pydantic",
        "httpx",
        "aiofiles",
        "python-multipart",
    ]

    DEFAULT_APT_PACKAGES = [
        "curl",
        "wget",
        "git",
        "sqlite3",
    ]

    def __init__(self, config: ModalConfig | None = None) -> None:
        """Initialize generator.

        Args:
            config: Modal configuration
        """
        self.config = config or ModalConfig()
        self.config.pip_packages = list(set(
            self.DEFAULT_PIP_PACKAGES + self.config.pip_packages
        ))
        self.config.apt_packages = list(set(
            self.DEFAULT_APT_PACKAGES + self.config.apt_packages
        ))

    def generate_app_py(self) -> str:
        """Generate main Modal app.py file.

        Returns:
            Python code as string
        """
        return f'''"""Lya Modal App - Autonomous Agent Deployment.

Generated for Python 3.14+ compatibility.
"""

import modal
import asyncio
from datetime import datetime, timezone
from typing import AsyncGenerator

# ═══════════════════════════════════════════════════════════════════════
# IMAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

image = (
    modal.Image.debian_slim(python_version="3.14")
    .apt_install({self.config.apt_packages!r})
    .pip_install({self.config.pip_packages!r})
    .env({self.config.environment!r})
)

# ═══════════════════════════════════════════════════════════════════════
# VOLUME FOR PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════

lya_volume = modal.Volume.from_name("lya-data", create_if_missing=True)

# ═══════════════════════════════════════════════════════════════════════
# APP CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

app = modal.App(
    name={self.config.app_name!r},
    image=image,
)

# ═══════════════════════════════════════════════════════════════════════
# LYA AGENT CLASS
# ═══════════════════════════════════════════════════════════════════════

class LyaAgent:
    """Lya agent running in Modal."""

    def __init__(self):
        self.started_at = datetime.now(timezone.utc)
        self.status = "initializing"
        self.working_memory = {{}}
        self.session_id = None

    async def start(self):
        """Start the agent."""
        # Initialize Lya core
        from lya.infrastructure.tools.direct_access import DirectAccess
        from lya.infrastructure.tools.tool_registry import get_tool_registry

        self.direct_access = DirectAccess(workspace="/mnt/lya-data")
        self.tools = get_tool_registry()

        self.status = "running"
        self.session_id = datetime.now(timezone.utc).isoformat()

        return {{
            "status": "started",
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
        }}

    async def process(self, message: str, context: dict | None = None) -> dict:
        """Process a message."""
        # Simple processing - integrate with actual Lya logic
        return {{
            "status": "processed",
            "message": message,
            "response": f"Received: {{message}}",
            "session_id": self.session_id,
        }}

    async def execute_tool(self, tool_name: str, params: dict) -> dict:
        """Execute a tool."""
        # Use direct access for immediate execution
        if tool_name == "file_read":
            return self.direct_access.read_file(params.get("path", ""))
        elif tool_name == "file_write":
            return self.direct_access.write_file(
                params.get("path", ""),
                params.get("content", "")
            )
        elif tool_name == "http_get":
            return self.direct_access.http_get(params.get("url", ""))
        elif tool_name == "execute_command":
            return self.direct_access.execute(params.get("command", ""))
        else:
            # Use registry for other tools
            return await self.tools.execute(tool_name, params)

    def get_status(self) -> dict:
        """Get agent status."""
        return {{
            "status": self.status,
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "uptime_seconds": (
                datetime.now(timezone.utc) - self.started_at
            ).total_seconds(),
        }}


# ═══════════════════════════════════════════════════════════════════════
# MODAL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

@app.function(
    gpu={self.config.gpu!r},
    cpu={self.config.cpu!r},
    memory={self.config.memory!r},
    timeout={self.config.timeout!r},
    volumes={{"/mnt/lya-data": lya_volume}},
    concurrency_limit={self.config.concurrency_limit!r},
)
@modal.web_endpoint(method="POST")
async def process_message(request: dict) -> dict:
    """Process a message from user."""
    agent = LyaAgent()
    await agent.start()

    message = request.get("message", "")
    context = request.get("context", {{}})

    result = await agent.process(message, context)
    return result


@app.function(
    gpu={self.config.gpu!r},
    cpu={self.config.cpu!r},
    memory={self.config.memory!r},
    timeout={self.config.timeout!r},
    volumes={{"/mnt/lya-data": lya_volume}},
)
@modal.web_endpoint(method="POST")
async def execute_tool(request: dict) -> dict:
    """Execute a tool."""
    agent = LyaAgent()
    await agent.start()

    tool_name = request.get("tool", "")
    params = request.get("params", {{}})

    result = await agent.execute_tool(tool_name, params)
    return result


@app.function(
    gpu={self.config.gpu!r},
    cpu={self.config.cpu!r},
    memory={self.config.memory!r},
    timeout={self.config.timeout!r},
    volumes={{"/mnt/lya-data": lya_volume}},
)
@modal.web_endpoint(method="GET")
async def health_check() -> dict:
    """Health check endpoint."""
    return {{
        "status": "healthy",
        "service": "lya-agent",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }}


@app.function(
    gpu={self.config.gpu!r},
    cpu={self.config.cpu!r},
    memory={self.config.memory!r},
    timeout={self.config.timeout!r},
    volumes={{"/mnt/lya-data": lya_volume}},
)
@modal.web_endpoint(method="GET")
async def agent_status() -> dict:
    """Get agent status."""
    agent = LyaAgent()
    await agent.start()
    return agent.get_status()


# ═══════════════════════════════════════════════════════════════════════
# WEBSOCKET SUPPORT (for real-time communication)
# ═══════════════════════════════════════════════════════════════════════

@app.function(
    gpu={self.config.gpu!r},
    cpu={self.config.cpu!r},
    memory={self.config.memory!r},
    timeout={self.config.timeout!r},
    volumes={{"/mnt/lya-data": lya_volume}},
    keep_warm=1,  # Keep warm for fast response
)
@modal.asgi_app()
def websocket_app():
    """WebSocket application for real-time communication."""
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse

    fastapi_app = FastAPI(title="Lya WebSocket API")

    class ConnectionManager:
        def __init__(self):
            self.active_connections: list[WebSocket] = []

        async def connect(self, websocket: WebSocket):
            await websocket.accept()
            self.active_connections.append(websocket)

        def disconnect(self, websocket: WebSocket):
            self.active_connections.remove(websocket)

        async def send_personal_message(self, message: str, websocket: WebSocket):
            await websocket.send_text(message)

        async def broadcast(self, message: str):
            for connection in self.active_connections:
                await connection.send_text(message)

    manager = ConnectionManager()
    agent = LyaAgent()

    @fastapi_app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await manager.connect(websocket)
        await agent.start()

        try:
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)

                # Process message
                result = await agent.process(
                    message.get("content", ""),
                    message.get("context", {{}})
                )

                await websocket.send_json(result)

        except WebSocketDisconnect:
            manager.disconnect(websocket)

    @fastapi_app.get("/health")
    async def health():
        return {{"status": "healthy"}}

    @fastapi_app.get("/status")
    async def status():
        return agent.get_status()

    return fastapi_app


# ═══════════════════════════════════════════════════════════════════════
# SCHEDULED JOBS (Background tasks)
# ═══════════════════════════════════════════════════════════════════════

@app.function(
    schedule=modal.Period(minutes=5),
    volumes={{"/mnt/lya-data": lya_volume}},
)
def self_improvement_job():
    """Run self-improvement every 5 minutes."""
    print("Running self-improvement check...")

    # Import and run self-improvement
    from lya.infrastructure.self_improvement.tool_generator import ToolGenerator

    generator = ToolGenerator()
    result = generator.identify_improvements()

    print(f"Improvement check complete: {{result}}")
    return result


@app.function(
    schedule=modal.Period(hours=1),
    volumes={{"/mnt/lya-data": lya_volume}},
)
def memory_consolidation_job():
    """Consolidate memories every hour."""
    print("Running memory consolidation...")

    # Import and run memory consolidation
    from lya.infrastructure.persistence.chroma_memory_repo import ChromaMemoryRepository

    repo = ChromaMemoryRepository()
    result = repo.consolidate_memories()

    print(f"Memory consolidation complete: {{result}}")
    return result


# ═══════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Lya Modal App")
    print("Run with: modal run app.py")
    print("Deploy with: modal deploy app.py")
'''

    def generate_requirements(self) -> str:
        """Generate requirements.txt for Modal.

        Returns:
            Requirements file content
        """
        lines = [
            "# Lya Modal Deployment Requirements",
            "# Generated for Python 3.14+",
            "",
            "# Core dependencies",
            "modal==0.63.*",
            "",
            "# Lya dependencies",
        ]

        for pkg in sorted(self.config.pip_packages):
            lines.append(pkg)

        lines.extend([
            "",
            "# Development (optional)",
            "pytest",
            "pytest-asyncio",
            "black",
            "ruff",
        ])

        return "\n".join(lines)

    def generate_modal_toml(self) -> str:
        """Generate modal.toml configuration.

        Returns:
            TOML configuration as string
        """
        return f'''# Modal Configuration for Lya

[app]
name = {self.config.app_name!r}

[environment]
PYTHON_VERSION = "3.14"
LYA_ENVIRONMENT = "modal"
LYA_DATA_PATH = "/mnt/lya-data"
'''

    def generate_deploy_script(self) -> str:
        """Generate deploy script.

        Returns:
            Bash script content
        """
        return '''#!/bin/bash
# Deploy Lya to Modal

echo "🚀 Deploying Lya to Modal..."

# Check if modal is installed
if ! command -v modal &> /dev/null; then
    echo "Installing Modal..."
    pip install modal
fi

# Authenticate (requires MODAL_TOKEN_ID and MODAL_TOKEN_SECRET)
if [ -z "$MODAL_TOKEN_ID" ] || [ -z "$MODAL_TOKEN_SECRET" ]; then
    echo "❌ Error: MODAL_TOKEN_ID and MODAL_TOKEN_SECRET must be set"
    echo "Get your tokens from: https://modal.com/settings/tokens"
    exit 1
fi

# Setup token
modal token set --token-id "$MODAL_TOKEN_ID" --token-secret "$MODAL_TOKEN_SECRET"

# Create volume for persistence
echo "Creating volume..."
modal volume create lya-data --env=main || true

# Deploy the app
echo "Deploying app..."
modal deploy app.py

echo "✅ Deployment complete!"
echo "View at: https://modal.com/apps"
'''

    def generate_client_py(self) -> str:
        """Generate client library for interacting with deployed Lya.

        Returns:
            Python client code
        """
        return '''"""Lya Modal Client.

Client library for interacting with Lya deployed on Modal.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any


@dataclass
class LyaClientConfig:
    """Configuration for Lya client."""
    base_url: str
    api_key: str | None = None
    timeout: int = 30


class LyaModalClient:
    """Client for Lya Modal deployment."""

    def __init__(self, config: LyaClientConfig):
        """Initialize client.

        Args:
            config: Client configuration
        """
        self.config = config
        self.base_url = config.base_url.rstrip("/")

    def _request(
        self,
        method: str,
        endpoint: str,
        data: dict | None = None
    ) -> dict[str, Any]:
        """Make HTTP request."""
        url = f"{self.base_url}{endpoint}"

        headers = {
            "Content-Type": "application/json",
        }

        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        try:
            if method == "GET":
                req = urllib.request.Request(
                    url,
                    headers=headers,
                    method="GET"
                )
            else:
                json_data = json.dumps(data or {}).encode()
                req = urllib.request.Request(
                    url,
                    data=json_data,
                    headers=headers,
                    method=method
                )

            with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                return json.loads(resp.read().decode())

        except urllib.error.HTTPError as e:
            return {
                "success": False,
                "error": f"HTTP {e.code}: {e.reason}",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def send_message(self, message: str, context: dict | None = None) -> dict[str, Any]:
        """Send message to Lya."""
        return self._request("POST", "/process_message", {
            "message": message,
            "context": context or {},
        })

    def execute_tool(self, tool: str, params: dict) -> dict[str, Any]:
        """Execute a tool."""
        return self._request("POST", "/execute_tool", {
            "tool": tool,
            "params": params,
        })

    def health_check(self) -> dict[str, Any]:
        """Check health status."""
        return self._request("GET", "/health_check")

    def get_status(self) -> dict[str, Any]:
        """Get agent status."""
        return self._request("GET", "/agent_status")


# Convenience function
def create_client(base_url: str, api_key: str | None = None) -> LyaModalClient:
    """Create Lya client."""
    return LyaModalClient(LyaClientConfig(base_url=base_url, api_key=api_key))
'''

    def generate_all(self, output_dir: str | Path) -> dict[str, str]:
        """Generate all deployment files.

        Args:
            output_dir: Directory to write files

        Returns:
            Dict of filename to content
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        files = {
            "app.py": self.generate_app_py(),
            "requirements.txt": self.generate_requirements(),
            "modal.toml": self.generate_modal_toml(),
            "deploy.sh": self.generate_deploy_script(),
            "client.py": self.generate_client_py(),
            "README.md": self._generate_readme(),
        }

        # Write files
        for filename, content in files.items():
            filepath = output_path / filename
            filepath.write_text(content, encoding="utf-8")

            # Make deploy.sh executable
            if filename == "deploy.sh":
                import stat
                filepath.chmod(filepath.stat().st_mode | stat.S_IEXEC)

        return {k: str(output_path / k) for k in files.keys()}

    def _generate_readme(self) -> str:
        """Generate README for deployment."""
        return '''# Lya Modal Deployment

24/7 autonomous agent deployment on Modal.com

## Quick Start

1. Set Modal credentials:
   ```bash
   export MODAL_TOKEN_ID=your_token_id
   export MODAL_TOKEN_SECRET=your_token_secret
   ```

2. Deploy:
   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   ```

3. Get your endpoint URL from the Modal dashboard

## Usage

```python
from client import create_client

client = create_client("https://your-app-url.modal.run")

# Send message
result = client.send_message("Hello Lya!")
print(result)

# Execute tool
result = client.execute_tool("file_read", {"path": "/tmp/test.txt"})
print(result)

# Check health
health = client.health_check()
print(health)
```

## Endpoints

- `POST /process_message` - Send message to Lya
- `POST /execute_tool` - Execute a tool
- `GET /health_check` - Health check
- `GET /agent_status` - Get agent status
- `WS /ws` - WebSocket for real-time communication

## Configuration

Edit `modal.toml` to customize:
- GPU type (A100, A10G, T4)
- Memory allocation
- CPU cores
- Timeout settings
'''


def generate_modal_deployment(
    output_dir: str | Path = "./modal_deployment",
    **kwargs
) -> dict[str, str]:
    """Generate complete Modal deployment.

    Args:
        output_dir: Output directory
        **kwargs: Configuration options

    Returns:
        Dict of generated file paths
    """
    config = ModalConfig(**kwargs)
    generator = ModalDeploymentGenerator(config)
    return generator.generate_all(output_dir)


# ═══════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Generate deployment files
    files = generate_modal_deployment(
        output_dir="./modal_deployment",
        app_name="lya-production",
        gpu="T4",  # Use T4 GPU for LLM inference
        cpu=4.0,
        memory=8192,
        secrets=["OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
        environment={
            "LYA_LOG_LEVEL": "INFO",
            "LYA_MAX_TOKENS": "4000",
        },
    )

    print("Generated Modal deployment files:")
    for name, path in files.items():
        print(f"  {name}: {path}")
