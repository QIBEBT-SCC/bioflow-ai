"""Application services for managing the tool catalog."""

from typing import TYPE_CHECKING, cast

from pydantic import BaseModel
from redis.asyncio import Redis
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import bioflow_ai.crud.tools.image as image_crud
import bioflow_ai.crud.tools.tool as tool_crud
import bioflow_ai.crud.tools.tool_doc as doc_crud
import bioflow_ai.crud.tools.tool_group as group_crud
import bioflow_ai.crud.tools.tool_tag as tag_crud
from bioflow_ai.core.config import get_settings
from bioflow_ai.llm.context import BaseContext
from bioflow_ai.llm.models import get_long_context_model
from bioflow_ai.llm.vectorstores import MilvusVectorStore
from bioflow_ai.models.tools import ToolDefine, ToolGroup, ToolHelpDoc, ToolImage, ToolTag
from bioflow_ai.schemas.tool_config import FileMount, ToolType
from bioflow_ai.schemas.tools import (
    DockerToolCreate,
    DockerToolUpdate,
    DocUpdate,
    ImageCreate,
    ImageUpdate,
    ToolGroupCreate,
)
from bioflow_ai.services._transaction import committing
from bioflow_ai.services.tools import cache as tool_cache
from bioflow_ai.services.tools.execution import ImageCommandResult, execute_image_command
from bioflow_ai.services.tools.image_references import assign_image_refs, lock_and_get_all
from bioflow_ai.services.tools.index import index_tools, refresh_tool_index, remove_tool_index
from bioflow_ai.services.workflow.validation.passthrough import validate_file_mounts
from bioflow_ai.utils.identifiers import UUIDLike, as_uuid
from bioflow_ai.utils.llm.structured_output import json_output

if TYPE_CHECKING:
    from bioflow_ai.llm.llm_client import LLMMetadata

_IMAGE_NOT_FOUND = 'Tool image not found'
_HELP_DOCUMENT_NOT_FOUND = 'Tool help document not found'
_HELP_DOCUMENT_IMAGE_NOT_FOUND = 'Tool image associated with the help document not found'
_TOOL_NOT_FOUND = 'Tool not found'
_CREATED_TOOL_RELOAD_FAILED = 'Created tool could not be reloaded'
_UPDATED_TOOL_RELOAD_FAILED = 'Updated tool could not be reloaded'
_AI_CHECKED_TAG_NOT_CONFIGURED = 'AI Checked tag is not configured'

CHECK_SYSTEM = """You are a bioinformatics expert. You will be given the help documentation of a bioinformatics tool, along with a user's proposed parameter modification for a specific command. Your task is to determine whether the modification is valid and return the result.
If the user provides fabricated parameters or uses parameters incorrectly, the modification should also be rejected, with the correct usage pointed out.

## Input

The user's modification proposal contains the following three parts:

1. **Origin Command**: The original bioinformatics tool command before modification. Parts enclosed in square brackets `[]` are modifiable parameters; parts enclosed in parentheses `()` are non-modifiable parameters.
   - **Modifiable parameters**: Parameters that do not affect the command's core functionality in the default encapsulation. Users may freely modify or replace these.
   - **Non-modifiable parameters**: Parameters strictly tied to the command's functionality (especially its inputs and outputs) in the default encapsulation. Modifying these may alter the output or the overall behavior of the command.
   - If no parameters are specially marked in the Origin Command, it means no such parameters are defined in the original encapsulation.
   - Placeholder tokens such as `{{genome_dir}}`, `{{read1}}`, `{{output_dir}}`, etc. represent fixed file arguments and should be ignored entirely during validation.
2. **New Args**: The updated modifiable parameters provided by the user, which will fully replace the content currently inside the square brackets `[]`.
3. **Modify Reason**: The reason behind the user's proposed modification.

## Validation Logic

Read the tool documentation and analyze the user's new modification parameters to decide whether to approve the modification:
- If the modification includes parameters that do not actually exist (or incorrect parameter names), reject the modification. In the `feedback` field, state the correct parameter based on the user's original modification request, and attach an excerpt from the relevant parameter documentation.
- If the modification includes incorrect usage of parameters, reject the modification. In the `feedback` field, explain the correct usage of the relevant parameter and attach an excerpt from the documentation.
- If the parameters are correct but potentially conflict with **Non-modifiable parameters**, reject the modification and explain the situation in the `feedback` field.
- If the new modification parameters and their usage are both correct and do not conflict with **Non-modifiable parameters**, approve the modification. The `feedback` field may be left blank.

## Output
Your output must be a plain JSON string containing exactly the following two fields, with no additional characters:

- `approve`: Boolean. Indicates whether the modification is approved.
- `feedback`: String. When the modification is not approved, provide the specific reason, such as the correct parameter to use or the need to create a new encapsulation. When approved, this field can be left as an empty string.
"""


class ArgsCheckResult(BaseModel):
    approve: bool
    feedback: str


async def check_update_args(
    doc: str,
    complete_command: str,
    update_arg: str,
    reason: str,
    user_id: int,
) -> ArgsCheckResult:
    """检查工具参数更新是否合法."""
    llm = get_long_context_model()
    metadata = cast('LLMMetadata', llm.metadata)
    chain = json_output(
        llm,
        CHECK_SYSTEM,
        ArgsCheckResult,
        metadata=metadata,
        name='check_update_args',
    )

    input_msg = (
        f'<ToolDoc>\n{doc}\n<ToolDoc>\n\n'
        f'Origin Command: `{complete_command}`\nNew Args: `{update_arg}`\nModify Reason: {reason}'
    )

    return await chain.ainvoke(
        input_msg,
        context=BaseContext(user_id=user_id),
    )


class ToolCatalogError(Exception):
    """Base exception for tool-catalog application services."""


class ToolCatalogNotFoundError(ToolCatalogError):
    """A required tool-catalog resource does not exist."""


class ToolCatalogConfigurationError(ToolCatalogError):
    """Required catalog configuration is missing or inconsistent."""


class ToolCommandExecutionError(ToolCatalogError):
    """A command required for a catalog operation failed."""

    def __init__(self, command: str, result: ImageCommandResult) -> None:
        self.command = command
        self.result = result
        super().__init__(f'Command failed with exit code {result.exit_code}: {command}')


class ToolInUseError(ToolCatalogError):
    """A tool cannot be deleted while workflows or runs reference it."""

    def __init__(self, workflow_count: int, run_count: int) -> None:
        self.workflow_count = workflow_count
        self.run_count = run_count
        super().__init__(
            f'Tool is used by {workflow_count} workflow(s) and {run_count} historical run(s).',
        )


def create_group(session: Session, group_data: ToolGroupCreate) -> ToolGroup:
    """Create a tool group and its corresponding tag atomically."""
    with committing(session):
        group_tag = tag_crud.insert(session, ToolTag(name=group_data.name))
        group = group_crud.insert(
            session,
            ToolGroup(name=group_data.name, parent_id=group_data.parent_id, tag_id=group_tag.id),
        )
    session.refresh(group)
    return group


def resolve_image_reference(session: Session, image_ref: str) -> ToolImage | None:
    """Resolve an exact generated tool-image reference."""
    return image_crud.get_by_ref(session, image_ref)


def create_image(session: Session, image_data: ImageCreate) -> ToolImage:
    """Create a tool image."""
    with committing(session):
        images = lock_and_get_all(session)
        image = ToolImage(
            name=image_data.name,
            aliases=image_data.aliases,
            version=image_data.version,
            description=image_data.description,
            homepage=image_data.homepage,
            paper_link=image_data.paper_link,
            image=image_data.image,
        )
        assign_image_refs([*images, image])
        image_crud.insert(session, image)
    session.refresh(image)
    return image


def update_image(session: Session, image_uid: UUIDLike, image_update: ImageUpdate) -> ToolImage | None:
    """Update a tool image."""
    with committing(session):
        images = lock_and_get_all(session)
        image = image_crud.update(session, image_uid, image_update)
        if image is None:
            return None
        assign_image_refs(images)
    session.refresh(image)
    return image


async def delete_image(session: Session, image_uid: UUIDLike) -> bool:
    """Delete a tool image and its vector index entry."""
    with committing(session):
        images = lock_and_get_all(session)
        deleted = image_crud.delete(session, image_uid)
        if not deleted:
            return False
        assign_image_refs([image for image in images if image.uid != as_uuid(image_uid)])

    vectorstore = MilvusVectorStore(
        collection_name='ImageCollection',
        uri=get_settings().knowledge_base.MILVUS_URL,
    )
    await vectorstore.adelete(
        filter_expression='image_uid == {uid}',
        filter_params={'uid': str(image_uid)},
    )
    return True


def execute_catalog_image_command(
    session: Session,
    image_uid: UUIDLike,
    command: str,
    *,
    ensure_ascii: bool = True,
) -> ImageCommandResult:
    """Run an ad-hoc command in a persisted tool image."""
    image = image_crud.get(session, image_uid)
    if image is None:
        raise ToolCatalogNotFoundError(_IMAGE_NOT_FOUND)
    return execute_image_command(image.image, command, ensure_ascii=ensure_ascii)


def refresh_help_document(session: Session, doc_uid: UUIDLike) -> ToolHelpDoc:
    """Refresh a help document by executing its configured command."""
    doc = doc_crud.get(session, doc_uid)
    if doc is None:
        raise ToolCatalogNotFoundError(_HELP_DOCUMENT_NOT_FOUND)
    if doc.image is None:
        raise ToolCatalogNotFoundError(_HELP_DOCUMENT_IMAGE_NOT_FOUND)

    result = execute_image_command(doc.image.image, doc.help_command)
    if not result.output:
        raise ToolCommandExecutionError(doc.help_command, result)

    with committing(session):
        doc_crud.update(session, doc_uid, doc_update=DocUpdate(content=result.output))
    session.refresh(doc)
    return doc


def _check_file_mounts(file_mounts: list[FileMount], tool_type: ToolType) -> None:
    try:
        validate_file_mounts(file_mounts, tool_type)
    except ValueError as error:
        raise ToolCatalogConfigurationError(str(error)) from error


async def create_tool(session: Session, tool_data: DockerToolCreate) -> ToolDefine:
    """Create a Docker tool, reusing or generating its help document, then index it."""
    _check_file_mounts(tool_data.file_mounts, tool_data.tool_type)
    image = image_crud.get(session, tool_data.image_uid)
    if image is None:
        raise ToolCatalogNotFoundError(_IMAGE_NOT_FOUND)

    tags = list(tag_crud.get_all_by_id(session, tool_data.tag_ids))
    help_doc = doc_crud.get_by_command(session, tool_data.help_command, tool_data.image_uid)

    with committing(session):
        if help_doc is None:
            result = execute_image_command(image.image, tool_data.help_command)
            if not result.output:
                raise ToolCommandExecutionError(tool_data.help_command, result)
            help_doc = doc_crud.insert(
                session,
                ToolHelpDoc(
                    help_command=tool_data.help_command,
                    content=result.output,
                    image_uid=tool_data.image_uid,
                ),
            )

        tool = ToolDefine(
            name=tool_data.name,
            image_uid=tool_data.image_uid,
            description=tool_data.description,
            tool_type=tool_data.tool_type,
            group_id=tool_data.group_id,
            command_template=tool_data.command_template,
            dynamic_params=tool_data.dynamic_params,
            immutable_static_params=tool_data.immutable_static_params,
            modifiable_static_params=tool_data.modifiable_static_params,
            file_mounts=tool_data.file_mounts,
            tags=tags,
            help_doc_uid=help_doc.uid,
        )
        tool_crud.insert(session, tool)
        tool_uid = tool.uid

    await index_tools([tool])

    persisted_tool = tool_crud.get(session, tool_uid)
    if persisted_tool is None:
        raise ToolCatalogNotFoundError(_CREATED_TOOL_RELOAD_FAILED)
    return persisted_tool


async def update_tool(
    session: Session,
    redis: Redis,
    tool_uid: UUIDLike,
    tool_update: DockerToolUpdate,
) -> ToolDefine | None:
    """Update a tool definition and invalidate its cached projections."""
    tool = tool_crud.get(session, tool_uid)
    if tool is None:
        return None

    description_changed = 'description' in tool_update.model_fields_set and tool_update.description != tool.description
    _check_file_mounts(
        tool_update.file_mounts if tool_update.file_mounts is not None else tool.file_mounts,
        tool_update.tool_type if tool_update.tool_type is not None else tool.tool_type,
    )
    update_data = {
        field_name: getattr(tool_update, field_name)
        for field_name in tool_update.model_fields_set
        if field_name != 'tags'
    }
    for field_name, value in update_data.items():
        setattr(tool, field_name, value)

    if tool_update.tags is not None:
        existing_tag_ids = [tag.id for tag in tool_update.tags if tag.id != -1]
        tool.tags = list(tag_crud.get_all_by_id(session, existing_tag_ids))

    with committing(session):
        tool_crud.update(session, tool)

    await tool_cache.invalidate_tool_projections(redis, tool_uid)
    if description_changed:
        await refresh_tool_index(tool)
    return tool_crud.get(session, tool_uid)


async def mark_tool_as_ai_checked(
    session: Session,
    redis: Redis,
    tool_uid: UUIDLike,
) -> ToolDefine:
    """Replace the AI review-state tag and invalidate cached projections."""
    tool = tool_crud.get(session, tool_uid)
    if tool is None:
        raise ToolCatalogNotFoundError(_TOOL_NOT_FOUND)

    checked_tag = tag_crud.get_by_name(session, 'AI Checked')
    if checked_tag is None:
        raise ToolCatalogConfigurationError(_AI_CHECKED_TAG_NOT_CONFIGURED)

    tool.tags = [tag for tag in tool.tags if tag.name not in {'AI Checked', 'AI Unchecked'}]
    tool.tags.append(checked_tag)

    with committing(session):
        tool_crud.update(session, tool)

    await tool_cache.invalidate_tool_projections(redis, tool_uid)
    persisted_tool = tool_crud.get(session, tool_uid)
    if persisted_tool is None:
        raise ToolCatalogNotFoundError(_UPDATED_TOOL_RELOAD_FAILED)
    return persisted_tool


async def delete_tool(
    session: Session,
    redis: Redis,
    tool_uid: UUIDLike,
) -> bool:
    """Delete a tool, its vector index entry, and its cached projections."""
    workflow_count, run_count = tool_crud.get_usage_counts(session, tool_uid)
    if workflow_count or run_count:
        raise ToolInUseError(workflow_count, run_count)

    try:
        with committing(session):
            deleted = tool_crud.delete(session, tool_uid)
            if not deleted:
                return False
    except IntegrityError as error:
        workflow_count, run_count = tool_crud.get_usage_counts(session, tool_uid)
        if workflow_count or run_count:
            raise ToolInUseError(workflow_count, run_count) from error
        raise

    await tool_cache.invalidate_tool_projections(redis, tool_uid)
    await remove_tool_index(tool_uid)
    return True
