from pathlib import Path

from bioflow_ai.core.paths import BUILTIN_PRESETS_DIR, SKILLS_DIR


def _copy_missing_files(source_dir: Path, destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for source in source_dir.iterdir():
        target = destination_dir / source.name
        if source.is_dir():
            _copy_missing_files(source, target)
            continue

        content = source.read_bytes()
        try:
            output = target.open('xb')
        except FileExistsError:
            continue

        try:
            with output:
                output.write(content)
        except OSError:
            target.unlink(missing_ok=True)
            raise


def install_deployment_presets() -> None:
    """Populate the editable skills and Compose SQL mount from bundled defaults."""
    for name, destination in (('skills', SKILLS_DIR), ('sql', Path('presets/sql'))):
        source = BUILTIN_PRESETS_DIR / name
        if source.resolve() != destination.resolve():
            _copy_missing_files(source, destination)
