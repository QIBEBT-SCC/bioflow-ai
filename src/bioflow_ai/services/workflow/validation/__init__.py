"""The only public operation for workflow validation."""

from typing import Any

from bioflow_ai.services.workflow.validation.types import ValidationResult, WorkflowDocument


def validate_workflow(workflow: WorkflowDocument | Any) -> ValidationResult:
    """Validate an entire document and return validity plus all diagnostics."""
    from bioflow_ai.services.workflow.validation.core import run_rules

    document = workflow if isinstance(workflow, WorkflowDocument) else WorkflowDocument(graph=workflow)
    return run_rules(document.graph, context=document.context, catalog=document.catalog)


__all__ = ['ValidationResult', 'WorkflowDocument', 'validate_workflow']
