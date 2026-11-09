"""Isolated input copies for testing generated pass-through Tool configurations."""

import hashlib
from pathlib import Path
from shutil import copy2, copytree


def copy_path(source: Path, destination: Path) -> Path:
    """Copy a file or directory, following symlinks so the copy shares no data with its source."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        copytree(source, destination, dirs_exist_ok=True)
    else:
        copy2(source, destination)
    return destination


def copy_validation_paths(paths: dict[str, Path], output_root: Path) -> dict[str, Path]:
    """Isolate mutable test inputs while keeping files that share a parent (such as companions) together."""
    directory_roots = sorted((path for path in paths.values() if path.is_dir()), key=lambda path: len(path.parts))
    copies: dict[str, Path] = {}
    for name, source in paths.items():
        parent = next((root for root in directory_roots if source.is_relative_to(root)), source.parent)
        group = hashlib.sha256(str(parent).encode()).hexdigest()[:24]
        copies[name] = copy_path(source, output_root / '_validation_inputs' / group / source.relative_to(parent))
    return copies
