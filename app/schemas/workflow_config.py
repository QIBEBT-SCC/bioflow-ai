from enum import IntEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class Status(IntEnum):
    WAITING = 0
    RUNNING = 1
    ERROR = 2
    SUCCESS = 3


class WorkflowType(IntEnum):
    SUBMODULE = 0
    TEMPLATE = 1


class ExecutionScope(IntEnum):
    SAMPLE_LEVEL = 0
    PROJECT_LEVEL = 1


class ToolOutput(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    result: dict[str, Path | str] = Field(default_factory=dict)
    log: Path | None = None
    reports: dict[str, Path | str] = Field(default_factory=dict)
    sample_name: str | None = None

    def all_files(self) -> dict[str, Path | str]:
        """Return all declared output paths or path-like references."""
        files = (self.result | self.reports).copy()
        if self.log:
            files['log'] = self.log
        return files
