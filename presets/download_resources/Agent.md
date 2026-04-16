# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Management: pixi

All environment operations must be performed through pixi. The direct use of conda, pip, venv, or mamba for dependency management is prohibited.

- All dependencies are added via `pixi add`. **Do not manually edit dependency version fields.**
- If there are no special requirements, the default Python version is 3.12, and the default R version is 4.3.

## Project Structure

```
project/
├── pixi.toml
├── pixi.lock
├── raw_data/           # Raw files from upstream analysis
├── results/            # Final outputs (figures, tables)
├── temp/               # Temporary files generated during analysis
├── analysis/           # Code
│   ├── 01_cluster.py / .R
│   ├── 02_visulization.py / .R
│   └── ...
└── docs/               # Documentation for intermediate analysis and final result interpretation
    ├── how_to_run_cluster.md
    ├── ...
    └── results.md      # Result interpretation
```