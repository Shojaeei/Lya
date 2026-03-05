# Migration Guide: From Legacy to Professional Structure

## Current State → Target State

### Legacy Structure
```
lya/
├── core.py
├── goals.py
├── planner.py
├── executor.py
├── memory.py
├── models.py
├── tools/
│   ├── file_tools.py
│   ├── system_tools.py
│   └── web_tools.py
└── __init__.py
```

### New Structure
```
src/lya/
├── domain/
│   ├── models/           ← entities (Goal, Task, Agent, Memory)
│   ├── repositories/     ← interfaces (GoalRepo, TaskRepo, MemoryRepo)
│   └── services/         ← domain services (PlanningService, ReasoningService)
├── application/
│   ├── ports/
│   │   ├── incoming/     ← commands & queries
│   │   └── outgoing/     ← LLM, Memory, Tool ports
│   ├── commands/         ← command handlers
│   └── queries/          ← query handlers
├── infrastructure/
│   ├── persistence/      ← repo implementations
│   ├── llm/              ← LLM adapters
│   └── tools/            ← tool implementations
└── adapters/
    ├── cli/              ← CLI interface
    ├── api/              ← REST API
    └── telegram/         ← Telegram bot
```

## Migration Steps

### Phase 1: Domain Layer (Week 1)

#### 1.1 Create Domain Models
```python
# Move from: goals.py
# Move to: domain/models/goal.py

# Old
class Goal:
    def __init__(self, description):
        self.description = description
        self.completed = False

# New
class Goal:
    def __init__(self, goal_id, description, priority=GoalPriority.MEDIUM):
        self._id = goal_id or uuid4()
        self._description = description
        self._priority = priority
        self._status = GoalStatus.PENDING
        # ... rich behavior
```

#### 1.2 Extract Repository Interfaces
```python
# New: domain/repositories/goal_repo.py
from abc import ABC, abstractmethod

class GoalRepository(ABC):
    @abstractmethod
    async def get(self, goal_id: UUID) -> Goal | None: ...
```

#### 1.3 Create Domain Services
```python
# New: domain/services/planning_service.py
class PlanningService:
    def __init__(self, llm: LLMPort):
        self._llm = llm
```

### Phase 2: Application Layer (Week 2)

#### 2.1 Create Ports
```python
# New: application/ports/outgoing/llm_port.py
class LLMPort(Protocol):
    async def generate(self, prompt: str) -> str: ...
```

#### 2.2 Move Commands
```python
# Old: main_loop.py
# New: application/commands/create_goal.py

class CreateGoalCommand:
    async def execute(self, agent_id, description) -> UUID:
        goal = Goal(description=description)
        await self._repo.save(goal)
        await self._event_bus.publish(GoalCreated(...))
        return goal.id
```

### Phase 3: Infrastructure (Week 3)

#### 3.1 Implement Repositories
```python
# New: infrastructure/persistence/chroma_goal_repo.py
from lya.domain.repositories.goal_repo import GoalRepository

class ChromaGoalRepository(GoalRepository):
    def __init__(self, client: ChromaClient):
        self._client = client
```

#### 3.2 Create LLM Adapters
```python
# New: infrastructure/llm/ollama_adapter.py
from lya.application.ports.outgoing.llm_port import LLMPort

class OllamaAdapter(LLMPort):
    def __init__(self, base_url: str, model: str):
        self._client = httpx.AsyncClient(base_url=base_url)
        self._model = model
```

### Phase 4: Adapters (Week 4)

#### 4.1 Refactor CLI
```python
# Old: chat_interface.py
# New: adapters/cli/main.py

@click.command()
def run():
    handler = container.create_goal_handler()
    goal_id = asyncio.run(handler.execute(...))
```

#### 4.2 Create API
```python
# New: adapters/api/server.py
from fastapi import FastAPI

app = FastAPI()

@app.post("/goals")
async def create_goal(request: CreateGoalRequest):
    handler = container.create_goal_handler()
    goal_id = await handler.execute(...)
    return {"goal_id": goal_id}
```

## Code Transformation Examples

### Example 1: Goal Creation

**Before:**
```python
# main_loop.py
class MainLoop:
    def add_goal(self, description, priority=5):
        goal_id = f"goal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.goals[goal_id] = {
            "description": description,
            "priority": priority,
            "completed": False
        }
        return goal_id
```

**After:**
```python
# domain/models/goal.py
class Goal:
    def __init__(self, description: str, priority: GoalPriority):
        self._id = uuid4()
        self._description = description
        self._priority = priority
        self._status = GoalStatus.PENDING

    def complete(self, result: Any) -> None:
        self._status = GoalStatus.COMPLETED
        self._completed_at = datetime.utcnow()
        self._result = result

# application/commands/create_goal.py
class CreateGoalHandler:
    async def execute(self, agent_id: UUID, description: str,
                      priority: int = 3) -> UUID:
        goal = Goal(
            description=description,
            priority=GoalPriority(priority)
        )
        await self._repo.save(goal)
        await self._event_bus.publish(GoalCreated(...))
        return goal.id
```

### Example 2: Memory Storage

**Before:**
```python
# memory.py
class MemoryStore:
    def __init__(self, workspace):
        self.workspace = workspace
        self.memories = []

    def add(self, content):
        self.memories.append({
            "content": content,
            "timestamp": datetime.now()
        })
```

**After:**
```python
# domain/models/memory.py
class Memory:
    def __init__(self, content: str, memory_type: MemoryType):
        self._id = uuid4()
        self._content = content
        self._type = memory_type
        self._embedding: list[float] | None = None

# domain/repositories/memory_repo.py
class MemoryRepository(ABC):
    @abstractmethod
    async def search(self, query_embedding: list[float],
                     limit: int = 10) -> list[tuple[Memory, float]]: ...

# infrastructure/persistence/chroma_memory_repo.py
class ChromaMemoryRepository(MemoryRepository):
    async def search(self, query_embedding, limit=10):
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=limit
        )
        return [(Memory.from_dict(r), score) for r, score in results]
```

## Testing Migration

### Before
```python
# test_lya.py
def test_goal_creation():
    loop = MainLoop("/tmp/test")
    goal_id = loop.add_goal("Test goal")
    assert goal_id in loop.goals
```

### After
```python
# tests/unit/domain/test_goal.py
def test_goal_creation():
    goal = Goal(description="Test goal")
    assert goal.description == "Test goal"
    assert goal.status == GoalStatus.PENDING

# tests/unit/application/test_commands.py
async def test_create_goal_command():
    repo = MockGoalRepository()
    handler = CreateGoalHandler(repo)
    goal_id = await handler.execute(
        agent_id=uuid4(),
        description="Test goal"
    )
    assert await repo.get(goal_id) is not None
```

## Migration Checklist

### Week 1: Domain Layer
- [ ] Create domain model classes
- [ ] Add state machines to entities
- [ ] Create repository interfaces
- [ ] Write unit tests for domain

### Week 2: Application Layer
- [ ] Define ports (incoming/outgoing)
- [ ] Create command handlers
- [ ] Create query handlers
- [ ] Set up event bus
- [ ] Write application tests

### Week 3: Infrastructure
- [ ] Implement repository adapters
- [ ] Create LLM adapters
- [ ] Implement tools
- [ ] Add configuration
- [ ] Write integration tests

### Week 4: Adapters
- [ ] Refactor CLI
- [ ] Create API server
- [ ] Add WebSocket support
- [ ] Write E2E tests

### Week 5: DevOps
- [ ] Docker setup
- [ ] CI/CD pipeline
- [ ] Kubernetes manifests
- [ ] Documentation

## Common Pitfalls

1. **Don't** import infrastructure in domain
2. **Don't** use ORM models in domain
3. **Do** use domain events for communication
4. **Do** keep entities immutable where possible
5. **Do** use value objects for complex types

## Verification

After migration, verify:

```bash
# All tests pass
make test

# Type checking passes
make type-check

# Linting passes
make lint

# Application runs
make run

# Docker builds
docker-compose build
```
