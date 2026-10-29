# Repository development guide

This file provides guidance to coding agents working in this repository.

## Project Overview

BioFlowAI is a graph-based intelligent bioinformatics analysis platform that uses a visual node-based editor for designing and executing bioinformatics workflows. The system combines FastAPI backend,
Celery task queue, LangGraph AI agents, and containerized bioinformatics tools.

## Architecture

### Core Components

**Backend (FastAPI)**

- `src/bioflow_ai/runtime/server.py` - Main FastAPI application entry point
- `src/bioflow_ai/` - Installable application package organized by function:
    - `api/v1/endpoints/` - REST API endpoints
    - `models/` - SQLAlchemy database models
    - `schemas/` - Pydantic schemas for API validation
    - `crud/` - Database operations

**Task Queue (Celery)**

- `src/bioflow_ai/runtime/worker.py` - Celery worker entry points
- `src/bioflow_ai/runtime/celery_app.py` - Celery instance, discovery, and routing
- `src/bioflow_ai/tasks/` - Task implementations

**Package boundaries**

- `cli/` handles command parameters, prompts, presentation, and exception conversion.
- `runtime/` starts the API and workers; tasks must not import CLI modules.
- `services/` groups business logic by responsibility. Services accept presentation callbacks
  when needed and must not import CLI or terminal UI libraries.
- `infrastructure/` owns database, Redis, object storage, container runtimes, and external transport.
- `llm/` owns agents, tools, protocols, middleware, and workflow generation.
- `utils/` groups shared helpers by responsibility, including domain-specific subpackages.
- `core/paths.py` locates repository resources in root-level `presets/` and `deploy/`.
  Editable skills stay outside the wheel. Configuration and runtime output paths follow
  the existing working-directory and settings rules.

**Workflow System**

- Node-based visual editor (React Flow on frontend)
- Frozen execution plans dispatched through a generic Celery node task
- Supports: tool nodes, resource nodes, processor nodes, code nodes
- DAG-based execution with dependency resolution

### Key Patterns

**Node execution**

- `services/workflow/executors.py` maps executor names to task implementations and lists
  supported node types. `tasks/workflow.py` executes the prepared plan.
- Keep workflow compilation, prepared input contracts, and executor registration aligned
  when adding a node type.

**Project-Workflow-Sample Model**

- Projects contain samples (biological data)
- Workflows are templates that can be imported into projects
- Running a workflow on N samples creates N independent RunInstances
- Each RunInstance contains multiple TaskInstances (one per node)

## Project Environment

This project uses `uv` to manage the Python environment and dependencies.

- Use `uv sync` to install or update dependencies from `pyproject.toml` and `uv.lock`.
- Run Python modules, scripts, debugging commands, and tests through `uv run` so they execute inside the project environment.
- Prefer commands such as `uv run python ...`, `uv run pytest ...`, `uv run ruff check ...`, and `uv run bioflow-ai ...` over invoking global Python or globally installed tools directly.
- The equivalent module entry point is `uv run python -m bioflow_ai`.
- Run `uv run python -m pytest` for migration contract and offline service tests.
- See `MIGRATION.md` for the original module mapping, validation limits, and deployment cutover.

## Code Style

Project uses Ruff for linting and formatting:

- Line length: 120
- Quote style: single quotes
- Google-style docstrings
- Extensive lint rules enabled (see `pyproject.toml`)
