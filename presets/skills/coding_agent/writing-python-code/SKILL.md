---
name: writing-python-code
description: Create or edit script.py and the dependency list for a BioFlowAI code_python CODE definition, preserving its CLI and Python runtime contract.
---

# Write a Python CODE definition

## Authoring Files and Checks

- When editing an authoring workspace, deliver the script in `/workspace/script.py` and put one additional Python
  distribution requirement per line in `/workspace/dependencies.txt`, without installer flags or commands.
- Leave `dependencies.txt` empty when no additional dependency is needed.
- Inspect the generated source and CLI contract. Checks in biokit-utils alone do not establish compatibility with
  the target Python 3.13 runtime and its dependencies; report which checks actually ran.

## Target Runtime

- The script runs as one file through uv in a Python 3.13 container.
- Its current working directory is the writable node output directory.
- Upstream files and directories are mounted read-only and passed as named command-line options.
- Click is automatically provided by the runtime.
- The backend adds PEP 723 metadata; do not add it yourself.

## CLI and Files

- Define one required Click option for every input in the authoritative input contract.
- Use option names exactly as provided, including the leading `--`.
- Validate paths with `click.Path(exists=True, path_type=Path)`.
- Use `pathlib.Path` for filesystem operations.
- Write every result to the current working directory with a deterministic plain file name.
- Never modify or write beside upstream inputs.
- Include the standard `if __name__ == '__main__': main()` entry point.

## Dependencies

- Prefer the standard library when it keeps the implementation clear.
- Declare only additional Python distribution requirements.
- Do not declare Click.
- Use distribution names rather than import names, such as `PyYAML` instead of `yaml`.
- Do not use URL, Git, or local-path requirements.
- Do not create a virtual environment or invoke pip, uv, Bash, or R from the script.
