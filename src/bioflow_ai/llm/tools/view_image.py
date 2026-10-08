# ruff: noqa: TRY301 - validation errors are converted to recoverable image/tool failures.
"""Read a visual snapshot within the workflow assistant's evidence boundaries."""

from pathlib import Path

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from bioflow_ai.llm.context import WorkflowAssistantContext
from bioflow_ai.llm.tools.workflow_diagnosis import EvidenceSourceError, resolve_evidence_source
from bioflow_ai.services.agents.images import MAX_SOURCE_BYTES, ImagePreparationError, image_content, prepare_image


@tool('view_image', parse_docstring=True)
def view_image(
    source: str, runtime: ToolRuntime[WorkflowAssistantContext], path: str = '.', page: int = 1
) -> list[dict]:
    """View an image already available in the allowed workflow evidence filesystem.

    Args:
        source: run:current, verified run:child:<uid>, current sample:<tag>, or project proj:<keyword>.
        path: Relative image path below a directory source. Use '.' when the source is itself an image file.
        page: One-based TIFF page to inspect; defaults to the first page. Other formats only have page 1.
    """
    try:
        evidence = resolve_evidence_source(source, runtime.context)
        relative = Path(path)
        if relative.is_absolute() or '..' in relative.parts:
            message = 'Image path must be relative and remain within its evidence source.'
            raise ImagePreparationError(message)
        if evidence.kind == 'file':
            if path != '.':
                message = "Use path='.' for a file evidence source."
                raise ImagePreparationError(message)
            target = evidence.path
        else:
            target = (evidence.path / relative).resolve(strict=True)
            if not target.is_relative_to(evidence.path):
                message = 'Image path escapes its evidence source.'
                raise ImagePreparationError(message)
        if not target.is_file():
            message = 'The selected image path is not a regular file.'
            raise ImagePreparationError(message)
        runtime.stream_writer({'info': f'Reading image: {source}/{path}, page {page}'})
        with target.open('rb') as stream:
            image = prepare_image(stream.read(MAX_SOURCE_BYTES + 1), target.name, page)
        return [
            {'type': 'text', 'text': f'Filesystem image snapshot: source={source}, path={path}.'},
            *image_content(image),
        ]
    except (EvidenceSourceError, ImagePreparationError, OSError, ValueError) as error:
        return [{'type': 'text', 'text': f'Unable to view image: {error}'}]
