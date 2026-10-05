# Backend package migration

The migration runs in `codex/backend-src-layout`, starting at `1cfa198`.
The original checkout remains available for comparison. Runtime queues, Redis,
and production databases are not modified by the migration.

## Stages

- [x] 1. Capture CLI, OpenAPI, model, and static-check baselines.
- [x] 2. Move Python code into the installable `src/bioflow_ai` package.
- [x] 3. Separate process runtime and infrastructure from CLI commands.
- [x] 4. Group business services and LLM helpers by responsibility.
- [x] 5. Extract development, evaluation, and initialization business logic.
- [x] 6. Verify packaging, contracts, resources, and update documentation.

Each stage is committed separately. `tests/fixtures/layout_baseline.json` stores
the canonical contract digests for the original CLI, OpenAPI, and ORM tables.
`tests/fixtures/module_map.json` records destinations for every original Python
module, including modules split between CLI presentation and business services.
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
- At baseline, complete Celery discovery imported model configuration from PostgreSQL;
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

## Stage 5 verification

Business logic now lives in development, evaluation, initialization, and export
services. CLI prompts, Rich output, progress bars, and Click exception conversion
remain in the CLI package; services accept optional presentation callbacks.

- 18 offline tests pass, including actual discovery of all seven Celery tasks,
  unchanged queue routes, four checkpoint type round trips, preview subprocess
  execution, temporary DuckDB updates, fake-model completion/interruption/failure,
  editable skills, and atomic trajectory export failure handling.
- CLI/OpenAPI/ORM baseline digests match exactly.
- Pyrefly remains at 74 existing errors; migration introduces no new errors.
- No external model requests or production service mutations were performed.

## Final layout and verification

All 324 original modules have recorded destinations. The resulting package has
360 Python files, including new package initializers and extracted service/CLI
modules. No root-level `app`, `llm`, or `script` code directories remain.

```text
src/bioflow_ai/
  api/             core/             models/           schemas/          crud/
  services/        infrastructure/   tasks/            llm/              runtime/
  cli/             utils/
```

- `uv sync --frozen --extra podman` succeeds. Hatchling builds both sdist and wheel.
  The wheel contains only the `bioflow_ai` namespace and distribution metadata;
  the sdist includes the external SQL, skill, and deployment resources.
  Rebuilding a wheel from the extracted sdist also succeeds.
- Installing the wheel into a separate temporary environment and importing from
  outside the checkout succeeds. Root and first-level CLI help also pass there,
  with no editable source path present in `sys.path`.
- 21 offline tests pass. One opt-in isolated-service test passes separately;
  the normal test run skips it unless dedicated service URLs are supplied.
- The full CLI specification (including help, command names, parameter types,
  options, and defaults), OpenAPI, and ORM digests match the original baseline.
  All 83 API/model/schema/CRUD modules also retain identical AST definitions
  after excluding imports.
- Actual Celery discovery registers all seven expected tasks under `bioflow_ai`
  with unchanged queue purposes. Four moved checkpoint types round-trip offline.
  The preview worker runs as a real subprocess from a different directory.
- New temporary PostgreSQL 18 and Redis 8 containers verified the unchanged SQL
  initialization mount, administrator creation/persistence/login, authentication
  sessions, and a real Redis checkpoint round trip using the task serializer.
  Containers used random localhost ports and were removed after verification.
- Writable skill resources, Compose template lookup, preset imports outside the
  working directory, DuckDB record state transitions, evaluation resume inputs,
  container argument handling, and atomic trajectory export failure handling pass.
- Ruff reports 82 existing diagnostics, down from 89. Pyrefly reports the same
  74 existing errors. Comparing normalized diagnostics adds no new findings.
  Existing lint/type failures remain outside this migration's scope.
- `presets`, `deploy`, and `uv.lock` contents are unchanged. The original checkout
  remains clean and available for comparison.

Run the standard checks from the repository:

```bash
uv sync --frozen --extra podman
uv run python -m pytest
uv run python -m ruff check src tests
uv run python -m pyrefly check
uv build
```

To rerun the isolated-service test, create fresh dedicated PostgreSQL/Redis
instances first. Mount `presets/sql` read-only into PostgreSQL's initialization
directory, use database `bioflow_layout_test`, and bind both services to random
localhost ports. Supply `BIOFLOW_MIGRATION_POSTGRES_URL` and
`BIOFLOW_MIGRATION_REDIS_URL` only for those disposable services, then run
`uv run python -m pytest tests/test_isolated_services.py`. The test rejects the
usual production ports/database name. Remove the temporary services afterward.

Full workflow runs with real providers, Milvus/object storage, and production
deployment cutover were not executed. Agent execution checks use fake models;
no model requests, production queue cleanup, or checkpoint/data deletion occurs
as part of this migration.

## Build backend follow-up (2026-10-05)

The build backend is now `uv_build>=0.12.22,<0.13`, matching the installed uv
0.12.22 so its bundled backend can be used. Default discovery selects
`src/bioflow_ai`. Explicit source inclusion retains repository resources and
build context; generated preset data and local maintenance helpers
are excluded with recursive patterns.

Verification against the previous Hatchling distributions passes:

- `uv sync --frozen --extra podman`, `uv lock --check`, wheel/sdist builds,
  and rebuilding from an extracted sdist succeed; `uv.lock` is unchanged.
- All 357 wheel code files are byte-identical. Console entry points, Python
  requirements, optional extras, and runtime dependency metadata match.
- All 397 previous sdist files are retained, including SQL, editable skills,
  Compose templates, configuration examples, and repository documents.
  uv additionally retains the original build configuration as `pyproject.toml.orig`;
  the normalized `pyproject.toml` has identical parsed configuration.
- A separate temporary environment installs the wheel and imports outside the
  checkout without an editable source path. All 26 CLI help commands, the console
  entry point, `python -m bioflow_ai`, and the preview subprocess pass.
- Packaging probes confirm that generated Parquet/DuckDB files and local CLI
  maintenance code are excluded from both distributions.

Checks use temporary artifacts; previously removed migration tests and scripts
are not restored, and no production services or model providers are invoked.

The optional Chinese README, agent guides (`AGENTS.md` and `CLAUDE.md`), and
this migration record were subsequently removed from explicit source inclusion.
They remain in Git; installation and runtime do not require them.
