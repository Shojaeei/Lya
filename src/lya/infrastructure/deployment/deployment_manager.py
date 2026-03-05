"""Deployment Manager - Infrastructure Implementation."""

from __future__ import annotations

import asyncio
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from lya.domain.models.deployment import (
    DeploymentConfig, Deployment, BuildArtifact,
    DeploymentTarget, DeploymentStatus,
)
from lya.infrastructure.config.logging import get_logger
from lya.infrastructure.config.settings import settings

logger = get_logger(__name__)


class DeploymentManager:
    """
    Manages deployments to various targets.

    Supports: Docker, Docker Compose, Kubernetes, VPS
    """

    def __init__(self, workspace: Path | None = None):
        self.workspace = workspace or Path(settings.workspace_path)
        self.deployments_dir = self.workspace / "deployments"
        self.deployments_dir.mkdir(exist_ok=True)

        self._active_deployments: dict[UUID, Deployment] = {}

        logger.info("Deployment manager initialized")

    def create_dockerfile(self) -> Path:
        """Create optimized Dockerfile."""
        dockerfile_content = '''# Multi-stage build for Lya
FROM python:3.11-slim as builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production image
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY src/ ./src/
COPY pyproject.toml .
COPY README.md .
COPY CLAUDE.md .

# Install the package
RUN pip install --no-cache-dir -e .

# Environment
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV LYA_WORKSPACE=/data

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8080/health/')" || exit 1

# Expose port
EXPOSE 8080

# Data volume
VOLUME ["/data"]

# Run command
CMD ["python", "-m", "lya.adapters.api.main"]
'''
        dockerfile_path = self.workspace / "Dockerfile"
        with open(dockerfile_path, 'w', encoding='utf-8') as f:
            f.write(dockerfile_content)

        logger.info("Dockerfile created", path=str(dockerfile_path))
        return dockerfile_path

    def create_docker_compose(self) -> Path:
        """Create docker-compose.yml."""
        compose_content = '''version: '3.8'

services:
  lya:
    build: .
    image: lya:latest
    container_name: lya-agent
    ports:
      - "8080:8080"
    environment:
      - LYA_ENVIRONMENT=production
      - LYA_LOG_LEVEL=INFO
    volumes:
      - lya-data:/data
      - ./.env:/app/.env:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import httpx; httpx.get('http://localhost:8080/health/')"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Optional: ChromaDB for vector storage
  chroma:
    image: chromadb/chroma:latest
    container_name: lya-chroma
    volumes:
      - chroma-data:/chroma/chroma
    ports:
      - "8000:8000"
    environment:
      - IS_PERSISTENT=TRUE
      - PERSIST_DIRECTORY=/chroma/chroma

volumes:
  lya-data:
  chroma-data:
'''
        compose_path = self.workspace / "docker-compose.yml"
        with open(compose_path, 'w', encoding='utf-8') as f:
            f.write(compose_content)

        logger.info("Docker Compose file created", path=str(compose_path))
        return compose_path

    def create_kubernetes_yaml(self, config: DeploymentConfig | None = None) -> Path:
        """Create Kubernetes deployment manifests."""
        name = config.name if config else "lya"
        version = config.version if config else "latest"

        k8s_content = f'''apiVersion: apps/v1
kind: Deployment
metadata:
  name: {name}
  labels:
    app: {name}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: {name}
  template:
    metadata:
      labels:
        app: {name}
        version: {version}
    spec:
      containers:
      - name: lya
        image: lya:{version}
        ports:
        - containerPort: 8080
          name: http
        env:
        - name: LYA_ENVIRONMENT
          value: "production"
        - name: LYA_LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health/
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: data
          mountPath: /data
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: {name}-data
---
apiVersion: v1
kind: Service
metadata:
  name: {name}-service
spec:
  selector:
    app: {name}
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: {name}-data
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
'''
        k8s_path = self.workspace / f"k8s-{name}.yaml"
        with open(k8s_path, 'w', encoding='utf-8') as f:
            f.write(k8s_content)

        logger.info("Kubernetes manifest created", path=str(k8s_path))
        return k8s_path

    async def build_docker(
        self,
        tag: str = "lya:latest",
        push: bool = False,
        registry: str | None = None,
    ) -> dict[str, Any]:
        """Build Docker image."""
        self.create_dockerfile()

        deployment = Deployment(
            deployment_id=uuid4(),
            config=DeploymentConfig(
                config_id=uuid4(),
                target=DeploymentTarget.DOCKER,
                name="lya-docker-build",
                version=tag.split(":")[-1],
            ),
            status=DeploymentStatus.BUILDING,
            created_at=datetime.now(timezone.utc),
        )

        try:
            process = await asyncio.create_subprocess_exec(
                "docker", "build", "-t", tag, ".",
                cwd=str(self.workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                deployment.status = DeploymentStatus.SUCCESS
                result = {
                    "success": True,
                    "image": tag,
                    "deployment_id": str(deployment.deployment_id),
                    "logs": stdout.decode()[-500:] if stdout else "",
                }

                if push and registry:
                    await self._push_image(tag, registry)

            else:
                deployment.status = DeploymentStatus.FAILED
                result = {
                    "success": False,
                    "error": stderr.decode()[-500:] if stderr else "Build failed",
                }

            self._active_deployments[deployment.deployment_id] = deployment
            return result

        except Exception as e:
            deployment.status = DeploymentStatus.FAILED
            self._active_deployments[deployment.deployment_id] = deployment
            logger.error("Docker build failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def deploy_docker_local(
        self,
        port: int = 8080,
        tag: str = "lya:latest",
        name: str = "lya-agent",
    ) -> dict[str, Any]:
        """Deploy locally with Docker."""
        deployment = Deployment(
            deployment_id=uuid4(),
            config=DeploymentConfig(
                config_id=uuid4(),
                target=DeploymentTarget.DOCKER,
                name=name,
                version=tag.split(":")[-1],
            ),
            status=DeploymentStatus.DEPLOYING,
            created_at=datetime.now(timezone.utc),
        )

        try:
            # Stop existing container
            await asyncio.create_subprocess_exec(
                "docker", "rm", "-f", name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )

            # Start new container
            process = await asyncio.create_subprocess_exec(
                "docker", "run", "-d",
                "--name", name,
                "-p", f"{port}:8080",
                "-v", f"{self.workspace}/data:/data",
                "--restart", "unless-stopped",
                tag,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                container_id = stdout.decode().strip()[:12]
                deployment.status = DeploymentStatus.RUNNING
                deployment.started_at = datetime.now(timezone.utc)
                deployment.container_id = container_id
                deployment.endpoint = f"http://localhost:{port}"

                self._active_deployments[deployment.deployment_id] = deployment

                logger.info(
                    "Docker deployment successful",
                    container_id=container_id,
                    port=port,
                )

                return {
                    "success": True,
                    "container_id": container_id,
                    "port": port,
                    "url": f"http://localhost:{port}",
                    "deployment_id": str(deployment.deployment_id),
                }
            else:
                deployment.status = DeploymentStatus.FAILED
                self._active_deployments[deployment.deployment_id] = deployment
                return {"success": False, "error": stderr.decode()}

        except Exception as e:
            deployment.status = DeploymentStatus.FAILED
            self._active_deployments[deployment.deployment_id] = deployment
            logger.error("Docker deployment failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def deploy_to_kubernetes(
        self,
        config: DeploymentConfig | None = None,
    ) -> dict[str, Any]:
        """Deploy to Kubernetes."""
        config = config or DeploymentConfig(
            config_id=uuid4(),
            target=DeploymentTarget.KUBERNETES,
            name="lya",
            version="latest",
        )

        deployment = Deployment(
            deployment_id=uuid4(),
            config=config,
            status=DeploymentStatus.DEPLOYING,
            created_at=datetime.now(timezone.utc),
        )

        try:
            k8s_path = self.create_kubernetes_yaml(config)

            process = await asyncio.create_subprocess_exec(
                "kubectl", "apply", "-f", str(k8s_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                deployment.status = DeploymentStatus.RUNNING
                deployment.started_at = datetime.now(timezone.utc)

                self._active_deployments[deployment.deployment_id] = deployment

                return {
                    "success": True,
                    "output": stdout.decode(),
                    "deployment_id": str(deployment.deployment_id),
                }
            else:
                deployment.status = DeploymentStatus.FAILED
                self._active_deployments[deployment.deployment_id] = deployment
                return {"success": False, "error": stderr.decode()}

        except Exception as e:
            deployment.status = DeploymentStatus.FAILED
            self._active_deployments[deployment.deployment_id] = deployment
            logger.error("Kubernetes deployment failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def stop_deployment(self, deployment_id: UUID) -> bool:
        """Stop a running deployment."""
        deployment = self._active_deployments.get(deployment_id)
        if not deployment:
            return False

        try:
            if deployment.config.target == DeploymentTarget.DOCKER:
                process = await asyncio.create_subprocess_exec(
                    "docker", "rm", "-f", deployment.container_id or "lya-agent",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await process.wait()

            deployment.status = DeploymentStatus.STOPPED
            return True

        except Exception as e:
            logger.error("Failed to stop deployment", error=str(e))
            return False

    async def _push_image(self, tag: str, registry: str) -> dict[str, Any]:
        """Push image to registry."""
        try:
            # Tag for registry
            registry_tag = f"{registry}/{tag}"

            process = await asyncio.create_subprocess_exec(
                "docker", "tag", tag, registry_tag,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.communicate()

            # Push
            process = await asyncio.create_subprocess_exec(
                "docker", "push", registry_tag,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                return {"success": True, "image": registry_tag}
            else:
                return {"success": False, "error": stderr.decode()}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_active_deployments(self) -> list[Deployment]:
        """Get list of active deployments."""
        return [
            d for d in self._active_deployments.values()
            if d.status in (DeploymentStatus.RUNNING, DeploymentStatus.DEPLOYING)
        ]
