# Contributing to BioFlow AI

Thank you for contributing to BioFlow AI! This guide covers the local development setup, branching conventions, and release workflow.

## Development Environment

- **Python**: 3.13+
- **Package manager**: [uv](https://docs.astral.sh/uv/)

Install locked dependencies from a source checkout:

```bash
uv sync                 # Docker setup
uv sync --extra podman  # Podman setup
```

Run project commands and services through `uv run` so they use the managed environment:

```bash
uv run bioflow-ai --help
uv run python -m bioflow_ai --help
```

### Quality Checks and Code Style

```bash
uv run ruff check
uv run ruff format
uv run ty check
uv run python -m pytest
```

- **Formatting & linting**: Ruff (120-character line length, single quotes, Google-style docstrings; see [pyproject.toml](./pyproject.toml)).
- **CI checks**: Installation and packaging checks in [pr-ci.yml](./.github/workflows/pr-ci.yml) (`Backend checks`) are blocking. Ruff and `ty` currently run as advisory checks during rollout.

## Branches and Pull Requests

- `master` is the default branch and the source of releases. `dev` is the integration branch.
- Create `feature/<topic>` or `fix/<topic>` branches from `dev`. Open pull requests against `dev`, wait for **Backend checks** CI to pass, squash merge, and delete the topic branch.
- After integration testing, open a release pull request from `dev` to `master` and merge with a merge commit.
- For an urgent production fix, branch `hotfix/<topic>` from the deployed release, release the fix, and bring the fix back into both `master` and `dev` as needed.

## Releases

Backend and frontend versions are independent.

1. In the `dev` -> `master` release pull request, bump the PEP 440 version in `pyproject.toml` (for example, `1.0.0b1` or `1.0.0`) and refresh `uv.lock`:
   ```bash
   uv lock
   ```
2. After merging to `master`, trigger the **Release & Publish to PyPI** workflow ([release.yml](./.github/workflows/release.yml)) from the `master` branch in GitHub Actions (optionally run with `dry_run: true` first to verify without publishing).
3. The release workflow automates:
   - Deriving the matching SemVer Git tag from `pyproject.toml` (for example, `1.0.0b1` -> `v1.0.0-beta.1`) and validating `uv.lock`.
   - Building the sdist and wheel, validating bundled resources and metadata with `twine check --strict`, and smoke-testing the wheel in an isolated virtual environment.
   - Publishing the distributions to PyPI via Trusted Publishing (`pypi` environment).
   - Creating the GitHub Release with attached `.whl` and `.tar.gz` assets and generated release notes.
4. Afterwards, edit the generated GitHub Release notes to link related frontend and backend pull requests and record the tested frontend image along with any database or configuration changes.
