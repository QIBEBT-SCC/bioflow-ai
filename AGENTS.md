# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BioFlowAI is a graph-based intelligent bioinformatics analysis platform that uses a visual node-based editor for designing and executing bioinformatics workflows. The system combines FastAPI backend,
Celery task queue, LangGraph AI agents, and containerized bioinformatics tools.

## Architecture

### Core Components

**Backend (FastAPI)**

- `script/server.py` - Main FastAPI application entry point
- `app/` - Application code organized by function:
    - `api/v1/endpoints/` - REST API endpoints
    - `models/` - SQLModel database models
    - `schemas/` - Pydantic schemas for API validation
    - `crud/` - Database operations

**Task Queue (Celery)**

- `script/worker.py` - Celery worker entry point
- `app/tasks/` - Task implementations

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

**Project-Workflow-Sample Model**

- Projects contain samples (biological data)
- Workflows are templates that can be imported into projects
- Running a workflow on N samples creates N independent RunInstances
- Each RunInstance contains multiple TaskInstances (one per node)

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
