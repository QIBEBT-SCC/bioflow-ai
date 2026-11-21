"""Locate bundled resources and the writable presets used by this deployment."""

from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _PACKAGE_DIR.parent.parent
_IS_SOURCE_TREE = _PACKAGE_DIR.parent.name == 'src' and (_SOURCE_ROOT / 'pyproject.toml').is_file()
_RESOURCE_ROOT = _SOURCE_ROOT if _IS_SOURCE_TREE else _PACKAGE_DIR / '_resources'

BUILTIN_PRESETS_DIR = _RESOURCE_ROOT / 'presets'
PRESETS_DIR = BUILTIN_PRESETS_DIR if _IS_SOURCE_TREE else Path('presets')
SKILLS_DIR = PRESETS_DIR / 'skills'
DEPLOY_DIR = _RESOURCE_ROOT / 'deploy'
