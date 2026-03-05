# Lya Architecture Overview

## Philosophy

Lya follows **Clean Architecture** (also known as Hexagonal or Ports and Adapters architecture) to ensure:

1. **Independence of Frameworks** - Domain logic doesn't depend on FastAPI, SQLAlchemy, or any other framework
2. **Testability** - Business rules can be tested without UI, database, or external services
3. **Independence of UI** - The interface can be CLI, API, or Telegram without changing domain logic
4. **Independence of Database** - We can swap Chroma for Qdrant or add PostgreSQL without touching business rules
5. **Independence of External Services** - LLM providers can be changed without affecting core logic

## Architectural Layers

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ADAPTERS LAYER                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐   │
│  │    CLI     │  │    API     │  │  Telegram  │  │   Voice    │   │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                              │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                        Use Cases                              │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐   │   │
│  │  │  Commands  │  │  Queries   │  │   Event Handlers      │   │   │
│  │  │            │  │            │  │                       │   │   │
│  │  │ - Create   │  │ - Get      │  │ - GoalCompleted       │   │   │
│  │  │   Agent    │  │   Status   │  │ - TaskFailed          │   │   │
│  │  │ - Start    │  │ - Search   │  │ - ToolGenerated       │   │   │
│  │  │   Goal     │  │   Mem      │  │                       │   │   │
│  │  └────────────┘  └────────────┘  └────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                     PORTS (Interfaces)                        │   │
│  │  ┌────────────────────────────────────────────────────────┐   │   │
│  │  │              Incoming (Driven by External)               │   │   │
│  │  │    Commands, Queries                                    │   │   │
│  │  └────────────────────────────────────────────────────────┘   │   │
│  │  ┌────────────────────────────────────────────────────────┐   │   │
│  │  │              Outgoing (Driving External)               │   │   │
│  │  │    LLM, Memory, Tools                                   │   │   │
│  │  └────────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        DOMAIN LAYER                                 │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                        ENTITIES                               │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐              │   │
│  │  │   Agent    │  │    Goal    │  │    Task    │              │   │
│  │  │            │  │            │  │            │              │   │
│  │  │ - State    │  │ - Status   │  │ - Status   │              │   │
│  │  │ - Config   │  │ - Priority │  │ - Result   │              │   │
│  │  │ - Metrics  │  │ - Plan     │  │ - Retry    │              │   │
│  │  └────────────┘  └────────────┘  └────────────┘              │   │
│  │                                                              │   │
│  │  ┌────────────────────────────────────────────────────────┐   │   │
│  │  │                    VALUE OBJECTS                        │   │   │
│  │  │    Memory, Plan, TaskResult, MemoryContext               │   │   │
│  │  └────────────────────────────────────────────────────────┘   │   │
│  │                                                              │   │
│  │  ┌────────────────────────────────────────────────────────┐   │   │
│  │  │                     DOMAIN EVENTS                         │   │   │
│  │  │    GoalCreated, TaskCompleted, MemoryStored...         │   │   │
│  │  └────────────────────────────────────────────────────────┘   │   │
│  │                                                              │   │
│  │  ┌────────────────────────────────────────────────────────┐   │   │
│  │  │                   DOMAIN SERVICES                         │   │   │
│  │  │    PlanningService, ReasoningService                   │   │   │
│  │  └────────────────────────────────────────────────────────┘   │   │
│  │                                                              │   │
│  │  ┌────────────────────────────────────────────────────────┐   │   │
│  │  │                   REPOSITORY INTERFACES                 │   │   │
│  │  │    MemoryRepository, GoalRepository, TaskRepository    │   │   │
│  │  └────────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     INFRASTRUCTURE LAYER                            │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                      PERSISTENCE                              │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐              │   │
│  │  │   Chroma   │  │   Qdrant   │  │    File    │              │   │
│  │  │  Adapter   │  │  Adapter   │  │  Storage   │              │   │
│  │  └────────────┘  └────────────┘  └────────────┘              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                         LLM                                   │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐              │   │
│  │  │   Ollama   │  │  OpenAI    │  │  Anthropic │              │   │
│  │  │  Adapter   │  │  Adapter   │  │  Adapter   │              │   │
│  │  └────────────┘  └────────────┘  └────────────┘              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                        TOOLS                                │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐              │   │
│  │  │   File     │  │    Web     │  │  Browser   │              │   │
│  │  │   Tools    │  │   Tools    │  │   Tools    │              │   │
│  │  └────────────┘  └────────────┘  └────────────┘              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                     CONFIGURATION                             │   │
│  │    Settings, Logging, Security, Monitoring                   │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## Dependency Rule

Dependencies only point **inward**:

- **Domain** has NO dependencies on other layers
- **Application** depends only on Domain
- **Infrastructure** depends on Domain and Application
- **Adapters** depend on Domain and Application

## Directory Structure

```
lya/
├── domain/           # No external dependencies
│   ├── models/       # Entities, value objects
│   ├── repositories/ # Interfaces (not implementations)
│   ├── services/     # Domain services
│   └── exceptions.py
│
├── application/      # Depends only on domain
│   ├── ports/        # Interfaces
│   │   ├── incoming/  # What the app accepts (commands, queries)
│   │   └── outgoing/    # What the app needs (LLM, Memory, Tools)
│   ├── commands/     # Command handlers
│   ├── queries/      # Query handlers
│   └── events/       # Event handlers
│
├── infrastructure/   # Implements ports
│   ├── config/       # Settings, logging
│   ├── persistence/  # Chroma, Qdrant, File adapters
│   ├── llm/          # Ollama, OpenAI, Anthropic adapters
│   ├── tools/        # Tool implementations
│   ├── security/     # Sandboxing, permissions
│   └── messaging/    # Event bus
│
└── adapters/         # Interface adapters
    ├── cli/          # Command line interface
    ├── api/          # REST/WebSocket API
    ├── telegram/     # Telegram bot
    └── voice/        # Voice interface
```

## Key Principles

### 1. Dependency Inversion

```python
# Application layer defines what it needs (port)
class LLMPort(Protocol):
    async def generate(self, prompt: str) -> str: ...

# Infrastructure provides the implementation
class OllamaAdapter:
    async def generate(self, prompt: str) -> str:
        # Implementation details
        return await self._ollama.generate(prompt)

# Application doesn't know about Ollama
class PlanningService:
    def __init__(self, llm: LLMPort):  # Depends on abstraction
        self._llm = llm
```

### 2. Repository Pattern

Domain defines interface, infrastructure implements it:

```python
# Domain
class MemoryRepository(ABC):
    @abstractmethod
    async def get(self, memory_id: UUID) -> Memory | None: ...

# Infrastructure
class ChromaMemoryRepository(MemoryRepository):
    async def get(self, memory_id: UUID) -> Memory | None:
        # Chroma-specific implementation
```

### 3. CQRS (Command Query Responsibility Segregation)

Commands change state, queries only read:

```python
# Command - changes state
class CreateGoalCommand:
    async def execute(self, description: str) -> UUID:
        goal = Goal(description=description)
        await self._repo.save(goal)
        await self._event_bus.publish(GoalCreated(...))
        return goal.id

# Query - reads only
class GetGoalStatusQuery:
    async def execute(self, goal_id: UUID) -> dict:
        goal = await self._repo.get(goal_id)
        return goal.to_dict()
```

### 4. Event-Driven Architecture

Loose coupling through events:

```python
# Domain emits events
class Goal:
    def complete(self, result):
        self._status = GoalStatus.COMPLETED
        self._events.append(GoalCompleted(goal_id=self.id, result=result))

# Application handles events
class GoalCompletedHandler:
    async def handle(self, event: GoalCompleted):
        # Update metrics, notify user, trigger learning
```

## Testing Strategy

| Layer | Test Type | Scope |
|-------|-----------|-------|
| Domain | Unit | Business rules, entities |
| Application | Unit + Integration | Use cases, port implementations |
| Infrastructure | Integration | Database, API adapters |
| Adapters | E2E | Full flows via CLI/API |

## Migration Path

1. **Phase 1**: Create new structure alongside existing code
2. **Phase 2**: Migrate domain models (rich entities)
3. **Phase 3**: Add repositories (abstract storage)
4. **Phase 4**: Add application layer (use cases)
5. **Phase 5**: Migrate adapters (CLI, API)
6. **Phase 6**: Remove old code

## Benefits

1. **Testability**: Domain logic has no dependencies, easy to unit test
2. **Flexibility**: Swap LLM, database, or UI without changing business rules
3. **Maintainability**: Clear separation makes code easier to understand
4. **Scalability**: Distributed concerns are cleanly separated
5. **Team Organization**: Different teams can work on different layers
