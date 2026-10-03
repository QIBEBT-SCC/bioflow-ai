from pydantic import BaseModel, ConfigDict, Field

MAX_SKILL_CONTENT_LENGTH = 512_000
MAX_SKILL_RESOURCE_LENGTH = 1_000_000


class SkillSummary(BaseModel):
    """Skill metadata returned in an agent's catalog listing."""

    name: str
    description: str
    resource_count: int


class AgentSkillsPublic(BaseModel):
    """All skills assigned to one agent."""

    agent_name: str
    skills: list[SkillSummary]


class SkillCatalogPublic(BaseModel):
    """Complete administrator-visible skill catalog."""

    agents: list[AgentSkillsPublic]


class SkillPublic(BaseModel):
    """Complete editable representation of one skill."""

    agent_name: str
    name: str
    description: str
    content: str
    resources: list[str]


class SkillCreate(BaseModel):
    """Properties required to add a skill to an existing agent."""

    model_config = ConfigDict(extra='forbid')

    name: str = Field(min_length=1, max_length=64, pattern=r'^[a-z0-9]+(?:-[a-z0-9]+)*$')
    content: str = Field(min_length=1, max_length=MAX_SKILL_CONTENT_LENGTH)


class SkillUpdate(BaseModel):
    """Editable SKILL.md content."""

    model_config = ConfigDict(extra='forbid')

    content: str = Field(min_length=1, max_length=MAX_SKILL_CONTENT_LENGTH)


class SkillResourcePublic(BaseModel):
    """One editable UTF-8 resource bundled with a skill."""

    path: str
    content: str


class SkillResourceUpdate(BaseModel):
    """Text content used to add or replace a skill resource."""

    model_config = ConfigDict(extra='forbid')

    content: str = Field(max_length=MAX_SKILL_RESOURCE_LENGTH)
