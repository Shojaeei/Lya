# Lya Professional Project Structure

## Overview

This document describes the professional, production-grade architecture for the Lya Autonomous AGI Agent system. The architecture follows **Clean Architecture** principles with **Hexagonal Architecture** patterns.

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    ADAPTERS LAYER                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │    CLI      │  │  REST API   │  │  Telegram Bot      │ │
│  │  (Primary)  │  │  (Primary)  │  │  (Primary)          │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                  APPLICATION LAYER                        │
│  ┌─────────────────┐  ┌───────────────────────────────────┐  │
│  │   Commands      │  │           Queries                │  │
│  │  - CreateAgent  │  │  - GetAgentStatus               │  │
│  │  - AddGoal      │  │  - ListGoals                    │  │
│  │  - ExecuteTask  │  │  - SearchMemories               │  │
│  └─────────────────┘  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                  Event Handlers                        │  │
│  │         (GoalCompleted, TaskFailed, etc.)            │  │
│  └───────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    DOMAIN LAYER                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Domain Models (Aggregates)                │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │  │
│  │  │  Agent  │  │  Goal   │  │  Task   │  │ Memory  │   │  │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘   │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Domain Services                           │  │
│  │  ┌───────────────┐  ┌─────────────────────────────┐   │  │
│  │  │   Planning    │  │        Reasoning            │   │  │
│  │  │   Service     │  │        Service              │   │  │
│  │  └───────────────┘  └─────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │         Repository Interfaces (Ports)                │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │  │
│  │  │ GoalRepo    │  │  TaskRepo   │  │  MemoryRepo │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Domain Events                             │  │
│  │  GoalCreated • TaskCompleted • MemoryStored          │  │
│  └───────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                 INFRASTRUCTURE LAYER                      │
│  ┌───────────────────────────────────────────────────────┐  │
│  │            Repository Implementations                  │  │
│  │  ChromaMemoryRepo • FileGoalRepo • SQLiteTaskRepo     │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │           External Service Adapters                    │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐ │  │
│  │  │  LLM     │ │  Memory  │ │  Tools   │ │  Events │ │  │
│  │  │ Adapter  │ │ Adapter  │ │ Adapter  │ │  Bus    │ │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └─────────┘ │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │           Configuration & Utilities                  │  │
│  │  Settings • Logging • Security • Monitoring           │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Dependency Rule

```
┌────────────────────────────────────┐
│         ADAPTERS (UI/API)          │  ◄── Frameworks, Drivers
├────────────────────────────────────┤         ▲
│        APPLICATION LAYER           │         │
│    (Use Cases, Application Rules)  │         │
├────────────────────────────────────┤         │
│         DOMAIN LAYER               │         │
│   (Entities, Business Rules)     │         │
└────────────────────────────────────┘         │
              ▲                                │
              │                                │
              └────────────────────────────────┘
                    Dependencies point INWARD
```

## Directory Structure

```
lya/
├── .github/workflows/           # CI/CD pipelines
├── .docker/                    # Docker configuration
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── Dockerfile.dev
├── .kubernetes/                # K8s manifests
│   ├── base/
│   └── overlays/
├── docs/                       # Documentation
│   ├── architecture/           # ADRs, diagrams
│   ├── api/                    # API docs
│   ├── deployment/             # Deployment guides
│   └── development/            # Developer guides
├── src/lya/                    # Source code
│   ├── __init__.py
│   ├── domain/                 # Domain Layer
│   │   ├── models/             # Entities
│   │   ├── repositories/       # Repository interfaces
│   │   ├── services/           # Domain services
│   │   ├── events.py           # Domain events
│   │   └── exceptions.py       # Domain exceptions
│   ├── application/            # Application Layer
│   │   ├── ports/              # Interface contracts
│   │   │   ├── incoming/       # Driving ports (commands/queries)
│   │   │   └── outgoing/       # Driven ports (adapters)
│   │   ├── commands/           # Command handlers
│   │   ├── queries/            # Query handlers
│   │   └── events/             # Event handlers
│   ├── infrastructure/         # Infrastructure Layer
│   │   ├── config/             # Configuration
│   │   ├── persistence/        # Repository implementations
│   │   │   ├── vector/         # Chroma, Qdrant adapters
│   │   │   ├── file/           # File-based storage
│   │   │   └── cache/          # Caching layer
│   │   ├── llm/                # LLM provider adapters
│   │   ├── tools/              # Tool implementations
│   │   ├── security/           # Security components
│   │   └── messaging/          # Event bus, messaging
│   └── adapters/               # Interface Adapters
│       ├── cli/                # Command-line interface
│       ├── api/                # REST/WebSocket API
│       ├── telegram/           # Telegram bot
│       └── voice/              # Voice interface
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   ├── e2e/                    # End-to-end tests
│   └── fixtures/               # Test data
├── scripts/                    # Utility scripts
├── notebooks/                  # Jupyter notebooks
└── prototypes/                 # Experimental features
```

## Key Design Principles

### 1. Dependency Inversion

```python
# Domain layer defines ports (interfaces)
class GoalRepository(ABC):
    @abstractmethod
    async def get(self, goal_id: UUID) -> Goal | None: ...

# Infrastructure implements adapters
class ChromaGoalRepository(GoalRepository):
    async def get(self, goal_id: UUID) -> Goal | None:
        # Chroma-specific implementation

# Application depends on abstraction
class CreateGoalHandler:
    def __init__(self, repo: GoalRepository):  # Interface, not implementation
        self._repo = repo
```

### 2. Domain Events

```python
# Event definition in domain
@dataclass(frozen=True)
class GoalCompleted(DomainEvent):
    goal_id: str
    result: Any

# Publishing from entity
goal.complete(result)
event = GoalCompleted(goal_id=str(goal.id), result=result)
await event_bus.publish(event)

# Handling in application layer
@event_handler(GoalCompleted)
async def on_goal_completed(event: GoalCompleted):
    await memory.store(f"Completed goal: {event.result}")
    await metrics.increment("goals.completed")
```

### 3. Command/Query Separation (CQRS)

```python
# Commands modify state
class CreateGoalCommand:
    async def execute(agent_id, description) -> UUID:
        goal = Goal(description=description)
        await self._repo.save(goal)
        await self._event_bus.publish(GoalCreated(...))
        return goal.id

# Queries read state
class ListGoalsQuery:
    async def execute(agent_id, status=None) -> list[GoalDTO]:
        goals = await self._repo.get_by_agent(agent_id, status)
        return [GoalDTO.from_entity(g) for g in goals]
```

### 4. Repository Pattern

```python
# Domain defines interface
class MemoryRepository(ABC):
    @abstractmethod
    async def search(self, query_embedding, limit=10) -> list[Memory]: ...

# Infrastructure implements
class ChromaMemoryRepository(MemoryRepository):
    def __init__(self, client: ChromaClient):
        self._client = client

    async def search(self, query_embedding, limit=10) -> list[Memory]:
        results = self._client.query(embedding=query_embedding, n_results=limit)
        return [Memory.from_dict(r) for r in results]
```

## Configuration Management

```python
# Environment-based configuration
class Settings(BaseSettings):
    env: Literal["development", "staging", "production"] = "development"
    llm: LLMSettings
    memory: MemorySettings
    security: SecuritySettings

# Usage
settings = get_settings()
if settings.is_production:
    configure_production_logging()
```

## Testing Strategy

```
tests/
├── unit/                       # Fast, isolated tests
│   ├── domain/
│   │   ├── test_agent.py      # Business logic tests
│   │   ├── test_goal.py       # State machine tests
│   │   └── test_memory.py     # Entity tests
│   ├── application/
│   │   └── test_commands.py   # Use case tests
│   └── infrastructure/
│       └── test_adapters.py   # Adapter tests with mocks
│
├── integration/               # Slower, real dependencies
│   ├── test_llm_adapters.py   # Test with real LLM (mocked)
│   ├── test_memory_store.py   # Test with real Chroma
│   └── test_api.py            # Test API endpoints
│
└── e2e/                       # Full workflows
    └── test_agent_lifecycle.py # Complete scenarios
```

## Deployment

### Docker
```bash
# Development
docker-compose up

# Production
docker build -t lya:latest .
docker run -p 8000:8000 lya:latest
```

### Kubernetes
```bash
# Development
kubectl apply -k .kubernetes/overlays/development

# Production
kubectl apply -k .kubernetes/overlays/production
```

## Benefits of This Architecture

| Aspect | Benefit |
|--------|---------|
| **Testability** | Domain logic has no external dependencies - easy to unit test |
| **Flexibility** | Swap LLM providers, databases, or frameworks without changing business logic |
| **Maintainability** | Clear separation makes codebase easier to understand and modify |
| **Scalability** | Each layer can be scaled independently |
| **Team Parallelization** | Different teams can work on different layers simultaneously |

## Migration from Current Structure

### Phase 1: Foundation
1. Create new directory structure
2. Move domain models to `domain/models/`
3. Create repository interfaces
4. Add comprehensive type hints

### Phase 2: Application Layer
1. Implement command handlers
2. Create event handlers
3. Add query handlers
4. Implement event bus

### Phase 3: Infrastructure
1. Create adapter implementations
2. Add configuration management
3. Implement logging
4. Add security components

### Phase 4: Interfaces
1. Refactor CLI to use commands
2. Create API endpoints
3. Add WebSocket support
4. Implement Telegram bot

### Phase 5: DevOps
1. Add Docker configuration
2. Create CI/CD pipelines
3. Add Kubernetes manifests
4. Set up monitoring
