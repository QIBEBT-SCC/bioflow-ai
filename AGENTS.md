# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BioFlowAI is a graph-based intelligent bioinformatics analysis platform that uses a visual node-based editor for designing and executing bioinformatics workflows. The system combines FastAPI backend, Celery task queue, LangGraph AI agents, and containerized bioinformatics tools.

## Architecture

### Core Components

**Backend (FastAPI)**
- `api.py` - Main FastAPI application entry point
- `app/` - Application code organized by function:
  - `api/v1/endpoints/` - REST API endpoints
  - `models/` - SQLModel database models
  - `schemas/` - Pydantic schemas for API validation
  - `crud/` - Database operations
  - `utils/workflow/` - Workflow execution engine
  - `tasks/` - Celery task definitions

**AI Agents (LangGraph)**
- `llm/` - LangGraph-based AI agents:
  - `agents/reproduction/` - Workflow design agents (build_workflow_agent, managers)
  - `agents/tool_manager/` - Tool discovery and configuration agents
  - `agents/reporter/` - Result analysis agents
  - `tools/` - LangChain tools for agents (workflow, container, database, web_search)
  - `states.py` - State schemas for all agents
  - `skills/` - Markdown-based skill definitions loaded into agent context

**Task Queue (Celery)**
- `worker.py` - Celery worker entry point
- `app/tasks/` - Task implementations:
  - `run_tool.py` - Execute bioinformatics tools in containers
  - `load_tasks/` - Load data (files, genomes, sequences)
  - `package_result.py` - Package workflow results

**Workflow System**
- Node-based visual editor (React Flow on frontend)
- Task handler registration system using decorators (`@register_handler`)
- Supports: tool nodes, resource nodes, processor nodes, code nodes
- DAG-based execution with dependency resolution

### Key Patterns

**Task Handler Registration**
- Handlers in `app/utils/workflow/handlers.py` use `@register_handler('node_type')` decorator
- Each node type has a handler implementing `create_task_instance()` and `create_celery_task()`
- See `docs/task_handler_regist.md` for detailed documentation

**AI Agent Architecture**
- Agents use LangGraph with state management (MessagesState)
- Manager pattern: main agent delegates to specialized manager agents
- Middleware system for cross-cutting concerns (token stats, status bar, skills)
- Context objects (UserContext, BuildContext) pass user/session data

**Project-Workflow-Sample Model**
- Projects contain samples (biological data)
- Workflows are templates that can be imported into projects
- Running a workflow on N samples creates N independent RunInstances
- Each RunInstance contains multiple TaskInstances (one per node)

## Development Commands

### Setup and Configuration

```bash
# Copy example config and edit with your settings
cp config.example.toml config.toml

# Initialize database (creates tables, loads initial data)
python script/init_db.py

# Start podman service (required for container execution)
podman system service --time=0 unix:///run/user/$(id -u)/podman/podman.sock > /dev/null 2>&1 &
```

### Running Services

```bash
# Start API server (development)
python api.py

# Start Celery worker
python worker.py

# Or use Celery CLI directly
celery -A worker worker --loglevel=INFO
```

### Database Operations

```bash
# Initialize database with sample data
python script/init_db.py

# Build datasets for agent training
python script/build_datasets.py

# Export database
python script/export_db.py
```

### Testing

Tests are in `test/` directory. Run individual test files:

```bash
python test/test_workflow.py
python test/test_graph.py
python test/test_ai.py
```

## Configuration

Configuration uses TOML format (`config.toml`). Key sections:

- `[security]` - JWT settings
- `[network]` - Host, port, proxy settings
- `[persistent.database]` - PostgreSQL connection
- `[persistent.cache]` - Redis URLs for LLM cache, monitoring, API cache
- `[persistent.storage]` - Local storage paths
- `[persistent.storage.s3]` - MinIO/S3 configuration
- `[docker]` - Container runtime (docker/podman)
- `[celery]` - RabbitMQ broker and Redis backend
- `[knowledge_base]` - Milvus vector store and embedding models

## Code Style

Project uses Ruff for linting and formatting:
- Line length: 120
- Quote style: single quotes
- Google-style docstrings
- Extensive lint rules enabled (see `pyproject.toml`)

## Important Notes

**Container Runtime**
- System supports both Docker and Podman
- Container abstraction in `app/utils/container/`
- Tools execute in isolated containers with mounted volumes

**LLM Integration**
- Uses LangChain/LangGraph for agent orchestration
- Supports multiple providers (Anthropic, OpenAI, Google, Qwen)
- LLM cache uses Redis for response caching
- Token statistics tracked via middleware

**Workflow Execution**
- Workflows are DAGs converted to Celery task chains/groups
- Tasks execute in topological order respecting dependencies
- Task status tracked in database (WAITING → RUNNING → SUCCESS/ERROR)
- Results stored in user-specific directories under `storage/`

**Skills System**
- Skills are markdown files in `llm/skills/` loaded into agent context
- Provide domain-specific knowledge and examples
- Examples: CROSS_WORKFLOW_FILE_TRANSFER.md, MODIFY_WITH_AWK.md

**Adding New Node Types**
1. Create handler class in `app/utils/workflow/handlers.py`
2. Inherit from `BaseTaskHandler`
3. Add `@register_handler('node_type')` decorator
4. Implement `create_task_instance()` and `create_celery_task()`
5. Optionally override `handle()` for custom logic (e.g., incrementing task_index)

**Database Models**
- Use SQLModel (combines SQLAlchemy + Pydantic)
- Models in `app/models/`
- Relationships use foreign keys with cascade deletes
- UUID primary keys for most entities

**API Design**
- RESTful endpoints under `/api/v1/`
- Pagination support (offset/limit)
- User authentication via JWT tokens
- Dependency injection for database sessions (`SessionDep`) and users (`UserDep`)
