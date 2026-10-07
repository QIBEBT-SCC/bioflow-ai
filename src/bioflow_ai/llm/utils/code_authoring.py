"""CODE skill selection for structured code generation.

后续更新prebuild实现后需要删除.
"""

from bioflow_ai.schemas.code import CodeNodeType
from bioflow_ai.services.skills.definitions import SKILLS_DIR, load_skill_definition

LANGUAGE_SKILLS = {
    CodeNodeType.PYTHON: 'writing-python-code',
    CodeNodeType.R: 'writing-r-code',
    CodeNodeType.BASH: 'writing-bash-code',
}


def load_code_skill(node_type: CodeNodeType) -> str:
    """Read the current language skill for a structured code generation request."""
    return load_skill_definition(SKILLS_DIR / 'coding_agent' / LANGUAGE_SKILLS[node_type]).instructions
