"""Repository resources kept outside the installed Python source tree."""

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PRESETS_DIR = REPOSITORY_ROOT / 'presets'
SKILLS_DIR = PRESETS_DIR / 'skills'
DEPLOY_DIR = REPOSITORY_ROOT / 'deploy'
