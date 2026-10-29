"""Keep media references in history and expand images only for model requests."""

import asyncio
import base64
import re
from html import escape
from typing import Any

from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage
from sqlalchemy.orm import Session

from bioflow_ai.services.agents.chat_attachment import (
    hydrate_image_references,
    image_reference_content,
    prepare_image,
    save_image,
)
from bioflow_ai.services.resource.file import store_file_bytes

IMAGE_TAG = re.compile(r'<Image id="([0-9]+)"\s*/?>')


def _part_text(part: str | dict[str, Any]) -> str:
    if isinstance(part, str):
        return part
    return part.get('text', '') if part.get('type') == 'text' else ''


def rebuild_multimodal_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Expand image tags in Human/Tool request copies; PDF tags remain plain text.

    Both standalone and embedded tags are supported. Existing multimodal blocks,
    message IDs, tool-call IDs and metadata are preserved without mutating history.
    """
    image_ids = []
    for message in messages:
        if not isinstance(message, (HumanMessage, ToolMessage)):
            continue
        contents = message.content if isinstance(message.content, list) else [message.content]
        for part in contents:
            text = _part_text(part)
            image_ids.extend(IMAGE_TAG.findall(text))
    if not image_ids:
        return messages
    images = hydrate_image_references(image_ids)
    result = []
    for message in messages:
        if not isinstance(message, (HumanMessage, ToolMessage)):
            result.append(message)
            continue
        contents = message.content if isinstance(message.content, list) else [message.content]
        expanded = []
        changed = False
        for part in contents:
            text = _part_text(part)
            matches = list(IMAGE_TAG.finditer(text))
            if not matches:
                expanded.append(part)
                continue
            changed = True
            position = 0
            for match in matches:
                if match.start() > position:
                    expanded.append({'type': 'text', 'text': text[position : match.start()]})
                expanded.append(images[match[1]])
                position = match.end()
            if position < len(text):
                expanded.append({'type': 'text', 'text': text[position:]})
        result.append(message.model_copy(update={'content': expanded}) if changed else message)
    return result


async def filter_multimedia_message(message: HumanMessage) -> HumanMessage:
    """Persist legacy evaluation media as file refs without parsing PDF content."""
    if not isinstance(message.content, list):
        return message

    def filter_content() -> HumanMessage:
        from bioflow_ai.infrastructure.database import engine
        from bioflow_ai.services._transaction import committing

        contents = []
        with Session(engine) as session, committing(session):
            for part in message.content:
                if not isinstance(part, dict) or part.get('type') == 'text':
                    contents.append(part)
                    continue
                data = base64.b64decode(part['base64'], validate=True)
                filename = part.get('filename', 'file')
                if part.get('type') == 'image':
                    image = save_image(session, prepare_image(data, filename))
                    contents.extend(image_reference_content(image))
                elif part.get('type') == 'file':
                    media_type = part.get('mime_type', 'application/octet-stream')
                    file = store_file_bytes(session, data, filename, media_type, 'documents')
                    tag = 'PDF' if media_type == 'application/pdf' else 'File'
                    contents.append(
                        {'type': 'text', 'text': f'<{tag} id="{file.id}" filename="{escape(filename, quote=True)}"/>'}
                    )
                else:
                    error_message = f'Unsupported media type: {part.get("type")}'
                    raise ValueError(error_message)
        return message.model_copy(update={'content': contents})

    return await asyncio.to_thread(filter_content)
