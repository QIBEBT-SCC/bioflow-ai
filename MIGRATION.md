# Backend package migration

The migration runs in `codex/backend-src-layout`, starting at `1cfa198`.
The original checkout remains available for comparison. Runtime queues, Redis,
and production databases are not modified by the migration.

## Stages

- [x] 1. Capture CLI, OpenAPI, model, and static-check baselines.
- [x] 2. Move Python code into the installable `src/bioflow_ai` package.
- [x] 3. Separate process runtime and infrastructure from CLI commands.
- [x] 4. Group business services and LLM helpers by responsibility.
- [ ] 5. Extract development, evaluation, and initialization business logic.
- [ ] 6. Verify packaging, contracts, resources, and update documentation.

Each stage is committed separately. `tests/fixtures/layout_baseline.json` stores
the canonical contract digests for the original CLI, OpenAPI, and ORM tables.
Full baseline captures and diagnostic details are available during this migration
in `/tmp/bioflow-layout-baseline` and can be regenerated from the original commit.

## Baseline

- Python source files: 324 (`app`: 212, `llm`: 101, `script`: 11).
- CLI: root and every first-level command/group `--help` pass without a config.
- OpenAPI: 135 paths, 180 operations, 190 schema components.
- ORM: 36 tables; no database schema migration is planned.
- Ruff: 89 existing diagnostics.
- Pyrefly: 74 existing errors with the original checkout's dependency environment.
- No tracked `tests/` suite existed before this migration.
- Complete Celery discovery currently imports model configuration from PostgreSQL;
  offline registration tests must replace model loading before task discovery.

## Runtime contracts

HTTP endpoints, CLI command names/options/defaults, workflow behavior, table names,
and external data directories remain unchanged. Python import paths and Celery
task names move to the `bioflow_ai` namespace. Old checkpoints and queued tasks
are not supported by the new package.

Repository resources (`presets`, including editable skills, and `deploy`) remain
outside the package. Configuration and runtime output paths retain their existing
working-directory/configuration semantics. The supported installation is still
`git clone`, `uv sync`, and `uv run` from the repository.

## Deployment cutover

Stop new task submission and the dispatcher, finish or cancel existing runs with
the old application, and stop old workers. Switch the API, dispatcher, and workers
together, then start new runs/checkpoints. Do not let workers from both import
namespaces consume the same queues during cutover. Keep existing workflow/tool
records and runtime files. Queue or checkpoint cleanup is a separate operational
action, not an installation side effect.
