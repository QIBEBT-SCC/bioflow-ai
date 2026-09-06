import uuid
from typing import Literal

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, SecretStr, model_validator

from app.schemas.code import CodeNodeType

CodingAgentProvider = Literal['codex', 'opencode']
OpenCodeModelProvider = Literal[
    'opencode-go',
    'opencode',
    'anthropic',
    'openai',
    'google',
    'openrouter',
    'deepseek',
    'custom',
]


class OpenCodeSettings(BaseModel):
    """Non-secret OpenCode credential selection shown in administration settings."""

    configured: bool = False
    model_provider: OpenCodeModelProvider = 'opencode-go'
    base_url: str | None = None
    model_id: str | None = None


class OpenCodeCredentialsCreate(BaseModel):
    """Write-only credentials for one OpenCode model provider."""

    model_config = ConfigDict(extra='forbid')
    api_key: SecretStr = Field(min_length=1, max_length=20_000)
    model_provider: OpenCodeModelProvider = 'opencode-go'
    base_url: AnyHttpUrl | None = None
    model_id: str | None = Field(default=None, min_length=1, max_length=200, pattern=r'^\S+$')

    @model_validator(mode='after')
    def validate_custom_provider(self) -> 'OpenCodeCredentialsCreate':
        """Require the endpoint and advertised model only for a custom provider."""
        if self.model_provider == 'custom':
            if self.base_url is None:
                message = 'Base URL is required for a custom OpenCode provider'
                raise ValueError(message)
            if self.model_id is None:
                message = 'Model ID is required for a custom OpenCode provider'
                raise ValueError(message)
        elif self.base_url is not None or self.model_id is not None:
            message = 'Base URL and model ID are only valid for a custom OpenCode provider'
            raise ValueError(message)
        return self


class CodingAgentProviderAvailability(BaseModel):
    """Credential availability for one supported ACP adapter."""

    provider: CodingAgentProvider
    name: str
    available: bool


class CodingAgentAvailability(BaseModel):
    """Minimal feature-discovery response used by CODE forms."""

    available: bool
    provider: Literal['codex'] = 'codex'
    name: Literal['Codex'] = 'Codex'
    providers: list[CodingAgentProviderAvailability] = Field(default_factory=list)


class CodexLoginPublic(BaseModel):
    """One temporary administrator device-login attempt."""

    id: uuid.UUID
    status: Literal['starting', 'waiting', 'completed', 'failed', 'cancelled']


class CodeAgentSessionCreate(BaseModel):
    """Create a temporary page-scoped ACP session."""

    model_config = ConfigDict(extra='forbid')

    node_type: CodeNodeType
    provider: CodingAgentProvider = 'codex'


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


class CodexAgentSettings(BaseModel):
    """Site-wide native Codex settings applied to new sessions."""

    model_config = ConfigDict(extra='forbid')
    sandbox_mode: Literal['read-only', 'workspace-write', 'danger-full-access'] = 'workspace-write'
    web_search: Literal['live', 'cached', 'disabled'] = 'live'
    network_access: bool = True

    def codex_config(self) -> dict[str, object]:
        """Map editable settings to native Codex configuration keys."""
        return {
            'sandbox_mode': self.sandbox_mode,
            'approval_policy': 'never',
            'approvals_reviewer': 'user',
            'web_search': self.web_search,
            'sandbox_workspace_write': {'network_access': self.network_access},
        }


class CodeAgentConfigUpdate(BaseModel):
    """Forward a session option selected from the agent's advertised options."""

    model_config = ConfigDict(extra='forbid')
    config_id: str = Field(min_length=1)
    value: str
