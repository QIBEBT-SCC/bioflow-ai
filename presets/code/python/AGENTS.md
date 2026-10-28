# Global development guidelines for the Bioflow AI python code

## Monorepo structure

The workspace is a Python project managed by `uv`. BioFlowAI collects `main.py` and the dependency declarations in
`pyproject.toml` for the saved CODE node.

```text
workspace/
├── .venv/           # Project environment managed by uv
├── output/          # All files produced by the script, without subdirectories
├── .python-version  # Python version selected by uv
├── main.py          # All workflow logic, helper functions, and classes
├── AGENTS.md        # Instructions for this workspace
├── pyproject.toml   # Project metadata and dependency declarations
└── uv.lock          # Resolved dependency lockfile
```

## Development tools & commands

- `uv` – Fast Python package installer and resolver (replaces pip/poetry)
- `ruff` – Fast Python linter and formatter
- `ty` – Static type checking

### Environment and dependency management

Use `uv` for all environment and dependency operations in this monorepo. Do not invoke `pip`, `poetry`, or `conda`
directly.

- Let `uv` manage the interpreter and virtual environments — `uv sync` and `uv run` operate without manual
  `source .venv/bin/activate`. Do not create ad-hoc virtual environments outside the package directory.
- Add dependencies by `uv add` and use `uv remove` to remove dependencies. Let uv resolve compatible versions and update
  both `pyproject.toml` and `uv.lock`. Do not edit it yourself.
- Install dependencies explicitly through `uv sync`; never let them install implicitly.

## Coding Guidelines

### Single-file implementation

- Keep all executable workflow logic, helper functions, and classes in `main.py`.
- Do not create additional local `.py` files or Python packages, or import helper modules from the workspace.
- Use imports for the Python standard library and third-party packages installed through `uv`.

### CLI and Files

- Define one required Click option for every input.
- Use option names exactly as provided, including the leading `--`.
- Use `pathlib.Path` for filesystem operations.
- Write every generated file, including final results, intermediate files, logs, and plots, directly into `output` directory using a deterministic plain file name. Keep the directory flat: do not create subdirectories.
- Build output paths with `Path.cwd() / 'output' / 'filename.ext'`. Do not hardcode `/workspace` paths or create another subdirectory inside `output/` directory.
- Never modify or write beside upstream inputs.
- Include the standard `if __name__ == '__main__': main()` entry point.
