import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.code import CodeNodeType


class CodingAgentAvailability(BaseModel):
    """Minimal feature-discovery response used by CODE forms."""

    available: bool
    provider: Literal['codex'] = 'codex'
    name: Literal['Codex'] = 'Codex'


class CodingAgentLoginPublic(BaseModel):
    """One temporary administrator device-login attempt."""

    id: uuid.UUID
    status: Literal['starting', 'waiting', 'completed', 'failed', 'cancelled']


class CodeAgentSessionCreate(BaseModel):
    """Create a temporary page-scoped ACP session."""

    model_config = ConfigDict(extra='forbid')

    node_type: CodeNodeType


class CodeAgentSessionPublic(BaseModel):
    """Public state for an ephemeral ACP session."""

    id: uuid.UUID
    status: Literal['starting', 'ready', 'running', 'proposal', 'closed', 'failed']


class CodeAgentTurnCreate(BaseModel):
    """One prompt plus the editor baseline for an ACP turn."""

    model_config = ConfigDict(extra='forbid')

    prompt: str = Field(min_length=1, max_length=20_000)
    source: str = Field(max_length=1_000_000)
    dependencies: list[str] = Field(default_factory=list, max_length=32)


class CodeAgentProposalDecision(BaseModel):
    """Decision result returned after accepting or rejecting a proposal."""

    proposal_id: uuid.UUID
    decision: Literal['accepted', 'rejected']


class CodeAgentProposal(BaseModel):
    """Whole-turn source and dependency candidate."""

    id: uuid.UUID
    base_hash: str
    source: str
    dependencies: list[str]
    diff: str
    warnings: list[str] = Field(default_factory=list)


class CodeAgentEvent(BaseModel):
    """Normalized server-sent event persisted in Redis."""

    id: str
    type: str
    data: dict[str, object]
