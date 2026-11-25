from typing import Literal

from sqlalchemy.orm import Session

from bioflow_ai.crud.resource import sample as sample_crud
from bioflow_ai.infrastructure.database import engine
from bioflow_ai.schemas.workflow_config import ToolOutput


def set_value(
    *,
    value: str,
    value_type: Literal['string'],  # noqa: ARG001
    sample_uid: str | None = None,
) -> ToolOutput:
    """Return literal text or a metadata value from the current sample."""
    if value.startswith('sample:'):
        if sample_uid is None:
            message = f'Cannot resolve {value}: no current sample.'
            raise ValueError(message)
        key = value.removeprefix('sample:')
        with Session(engine) as session:
            sample = sample_crud.get(session, sample_uid)
            if sample is None:
                message = f'Sample not found: {sample_uid}'
                raise KeyError(message)
            metadata = sample.meta_data or {}
            if key not in metadata or metadata[key] is None:
                message = f'Sample {sample.sample_name!r} has no metadata field {key!r}'
                raise KeyError(message)
            metadata_value = metadata[key]
            value = str(metadata_value)
    return ToolOutput(result={'value': value})
