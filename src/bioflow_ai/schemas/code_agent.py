"""Public contracts for temporary native CODE editing sessions."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from bioflow_ai.schemas.code import CodeNodeType

CodeSessionStatus = Literal['ready', 'queued', 'running', 'cancelling', 'proposal', 'closed']


class CodeAgentInput(BaseModel):
    model_config = ConfigDict(extra='forbid')


class CodeAgentSessionCreate(CodeAgentInput):
    node_type: CodeNodeType


class CodeAgentBaseline(CodeAgentInput):
    source: str = Field(default='', max_length=1_000_000)
    dependencies: list[str] = Field(default_factory=list, max_length=32)


class CodeAgentTurnCreate(CodeAgentBaseline):
    prompt: str = Field(min_length=1, max_length=100_000)
    language: Literal['zh', 'en'] | None = None

    @field_validator('prompt')
    @classmethod
    def nonempty_prompt(cls, value: str) -> str:
        """Reject requests consisting only of whitespace."""
        if not value.strip():
            msg = 'A coding request is required.'
            raise ValueError(msg)
        return value


class CodeAgentProposalDecision(CodeAgentBaseline):
    decision: Literal['accept', 'reject']


class CodeAgentProposal(BaseModel):
    id: str
    turn_id: str
    baseline_hash: str
    source: str
    dependencies: list[str]
    diff: str
    warnings: list[str]


class CodeAgentTranscriptMessage(BaseModel):
    role: Literal['user', 'assistant']
    text: str
    turn_id: str


class CodeAgentToolActivity(BaseModel):
    call_id: str
    turn_id: str
    name: str
    status: Literal['running', 'completed', 'failed', 'cancelled']
    detail: str = ''
    output: str = ''


class CodeAgentSessionPublic(BaseModel):
    id: str
    node_type: CodeNodeType
    status: CodeSessionStatus
    turn_id: str | None = None
    transcript: list[CodeAgentTranscriptMessage] = Field(default_factory=list)
    tools: list[CodeAgentToolActivity] = Field(default_factory=list)
    proposal: CodeAgentProposal | None = None
    error: str | None = None
    last_event_id: str = '0-0'


class CodeAgentAvailability(BaseModel):
    available: bool
    model_name: str | None = None


class CodeAgentEvent(BaseModel):
    id: str
    type: str
    data: dict[str, Any]
