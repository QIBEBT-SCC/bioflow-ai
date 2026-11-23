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

**Workflow System**

- Node-based visual editor (React Flow on frontend)
- Frozen execution plans dispatched through a generic Celery node task
- Supports: tool nodes, resource nodes, processor nodes, code nodes
- DAG-based execution with dependency resolution
- `services/workflow/executors.py` maps executor names to task implementations and lists supported node types.

## Project Environment

This project uses `uv` to manage the Python environment and dependencies.

- Use `uv sync` to install or update dependencies from `pyproject.toml` and `uv.lock`.
- Run Python modules, scripts, debugging commands, and tests through `uv run` so they execute inside the project environment.
- Prefer commands such as `uv run python ...`, `uv run pytest ...`, `uv run ruff check ...`, and `uv run bioflow-ai ...` over invoking global Python or globally installed tools directly.
- The equivalent module entry point is `uv run python -m bioflow_ai`.
- Using `uv run ty ...` for static type checking.
- Run `uv run python -m pytest` for migration contract and offline service tests.

## Branches and releases

Follow the branch, pull request, and release workflow in [CONTRIBUTING.md](./CONTRIBUTING.md).

When asked to open or submit a pull request targeting `dev` (unless the user explicitly asks to pause before merging), complete the full PR lifecycle in one pass:

- After creating the PR (`gh pr create --base dev`), wait for required CI checks to finish (`gh pr checks <pr> --watch`).
- Once required checks pass, squash-merge and delete the topic branch (`gh pr merge <pr> --squash --delete-branch`).
- Switch back to `dev`, fast-forward from `origin/dev`, prune stale remote refs (`git fetch --prune`), and ensure the local topic branch is removed while preserving any unrelated uncommitted files.
- If any required CI check fails or merge is blocked, stop and report the failure instead of force-merging. Do not auto-merge `dev` -> `master` release PRs unless explicitly requested.

## Code Style

Project uses Ruff for linting and formatting, and `ty` for static type checking:

- Line length: 120
- Quote style: single quotes
- Google-style docstrings
- Extensive lint rules enabled (see `pyproject.toml`)

After modifying Python files, format and check the affected files and resolve any diagnostics introduced by your changes:

- `uv run ruff format <files>` — format code
- `uv run ruff check <files>` — lint checks
- `uv run ty check <files>` — static type checking
