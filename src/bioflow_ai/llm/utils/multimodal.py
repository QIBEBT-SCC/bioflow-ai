"""
Multimodal message processing utilities

前端ui message -ui_input_to_human_message()-> 带有完整base64附件的langchain HumanMessage
multimodal langchain HumanMessage -filter_multimedia_message()-> 过滤后的langchain HumanMessage

"""

import re
from pathlib import Path
from typing import TypedDict
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from loguru import logger

from bioflow_ai.llm.utils.ai_sdk.ui_message import FileUIPart, ReasoningUIPart, TextUIPart, UIMessage
from bioflow_ai.utils.file import calculate_base64_sha256
from bioflow_ai.utils.file.storage import download_file_as_base64, upload_file_from_base64
from bioflow_ai.utils.pdf import parse_pdf


class MediaContent(TypedDict):
    type: str
    mime_type: str
    base64: str
    filename: str


def ui_input_to_human_message(ui_message: UIMessage) -> HumanMessage:
    """Convert input from Vercel UIMessage to LangChain HumanMessage

    Iterate through UIMessage.parts, concatenate plain text parts into a single string;
    if files (images, PDFs, text) are included, construct the multimodal content structure required by LangChain (image/file + base64 data),
    and finally return the appropriate HumanMessage.content based on the content type.

    Args:
        ui_message (UIMessage): The UI message object submitted by the client, internally containing a list of UIMessagePart.

    Returns:
        HumanMessage: User message object compatible with LangChain, directly usable for model invocation.
    """

    contents = []
    is_multimodel = False

    for part in ui_message.parts:
        if isinstance(part, TextUIPart):
            contents.append({'type': 'text', 'text': part.text})
        elif isinstance(part, FileUIPart):
            is_multimodel = True
            _, encoded = part.url.split(',', 1)
            if 'image' in part.mediaType:
                contents.append(
                    {
                        'type': 'image',
                        'mime_type': part.mediaType,
                        'base64': encoded,
                        'filename': part.filename,
                    }
                )
            elif part.mediaType == 'application/pdf':
                contents.append(
                    {
                        'type': 'file',
                        'mime_type': part.mediaType,
                        'base64': encoded,
                        'filename': part.filename,
                    }
                )
            elif part.mediaType == 'text/plain':
                contents.append(
                    {
                        'type': 'file',
                        'mime_type': part.mediaType,
                        'base64': encoded,
                        'filename': part.filename,
                    }
                )
            else:
                pass

    content = contents if is_multimodel else '\n'.join(item['text'] for item in contents)

    return HumanMessage(content=content)


async def filter_multimedia_message(message: HumanMessage):
    if isinstance(message.content, list):
        filtered_content = []
        for content in message.content:
            content: MediaContent
            match content['type']:
                case 'text':
                    filtered_content.append(content)
                case 'image':
                    file_hash = calculate_base64_sha256(content['base64'])
                    image_key = upload_file_from_base64(content['base64'], content['filename'], file_hash, 'images')
                    filtered_content.append({'type': 'text', 'text': f'<Image id="{image_key}">'})
                case 'file':
                    pdf_key = await parse_pdf(content['base64'], content['filename'])
                    filtered_content.append({'type': 'text', 'text': f'<PDF id="{pdf_key}">'})

        new_message = message.model_copy(update={'content': filtered_content})
        return new_message
    else:
        return message


def rebuild_multimodal_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """重建多模态消息，从占位符中提取文件并恢复完整的多模态内容

    将 <Image id="xxx"> 和 <PDF id="xxx"> 占位符替换为完整的多模态消息结构

    Args:
        messages: 包含占位符的消息列表

    Returns:
        重建后的消息列表，包含完整的 base64 数据
    """
    # 正则表达式匹配占位符
    image_pattern = re.compile(r'<Image id="([^"]+)">')
    pdf_pattern = re.compile(r'<PDF id="([^"]+)">')

    rebuilt_messages = []

    for message in messages:
        if isinstance(message, HumanMessage) and isinstance(message.content, list):
            rebuilt_content = []

            for content_item in message.content:
                assert content_item.get('type') == 'text'
                text = content_item.get('text', '')

                # 检查是否包含图片占位符
                image_match = image_pattern.match(text)
                if image_match:
                    image_key = image_match.group(1)
                    filename = Path(image_key).name
                    try:
                        base64_data = download_file_as_base64(image_key)

                        ext = Path(filename).suffix.lower()
                        mime_type_map = {
                            '.png': 'image/png',
                            '.jpg': 'image/jpeg',
                            '.jpeg': 'image/jpeg',
                            '.gif': 'image/gif',
                            '.webp': 'image/webp',
                        }
                        mime_type = mime_type_map.get(ext, 'image/png')

                        rebuilt_content.append(
                            {
                                'type': 'image',
                                'mime_type': mime_type,
                                'base64': base64_data,
                                'filename': filename,
                            }
                        )
                        continue
                    except Exception as e:
                        logger.warning(f'Failed to rebuild image from key {image_key}: {e}')
                        rebuilt_content.append(content_item)
                        continue

                # 检查是否包含 PDF 占位符
                pdf_match = pdf_pattern.match(text)
                if pdf_match:
                    pdf_key = pdf_match.group(1)
                    filename = Path(pdf_key).name
                    try:
                        base64_data = download_file_as_base64(pdf_key)

                        rebuilt_content.append(
                            {
                                'type': 'file',
                                'mime_type': 'application/pdf',
                                'base64': base64_data,
                                'filename': filename,
                            }
                        )
                        continue
                    except Exception as e:
                        logger.warning(f'Failed to rebuild PDF from key {pdf_key}: {e}')
                        rebuilt_content.append(content_item)
                        continue

                rebuilt_content.append(content_item)

            rebuilt_message = message.model_copy(update={'content': rebuilt_content})
            rebuilt_messages.append(rebuilt_message)
        else:
            rebuilt_messages.append(message)

    return rebuilt_messages


def langchain_messages_to_ui_messages(messages: list[BaseMessage]) -> list[UIMessage]:
    """Convert LangChain message list to vercel UIMessage list

    Args:
        messages: LangChain messages

    Returns:
        UIMessage list
    """
    ui_messages = []

    ai_parts = []
    for lc_msg in messages:
        if isinstance(lc_msg, HumanMessage):
            if isinstance(lc_msg.content, list):
                human_parts = []
                for content in lc_msg.content:
                    if content['type'] == 'text':
                        human_parts.append(TextUIPart(text=content['text']))
                    else:
                        human_parts.append(
                            FileUIPart(
                                mediaType=content['type'],
                                filename=content['filename'],
                                url=f'data:{content["mime_type"]};base64,{content["base64"]}',
                            )
                        )
                ui_messages.append(
                    UIMessage(
                        role='user',
                        id=lc_msg.id if lc_msg.id else uuid4(),
                        parts=human_parts,
                    )
                )
            else:
                ui_messages.append(
                    UIMessage(
                        role='user',
                        id=lc_msg.id if lc_msg.id else uuid4(),
                        parts=[TextUIPart(text=lc_msg.content)],
                    )
                )
        else:
            if isinstance(lc_msg, AIMessage):
                if hasattr(lc_msg, 'additional_kwargs') and 'reasoning_content' in lc_msg.additional_kwargs:
                    reasoning_content = lc_msg.additional_kwargs['reasoning_content']
                    if reasoning_content:
                        ai_parts.append(ReasoningUIPart(text=reasoning_content))

                if lc_msg.content:
                    # 我们假设内容是纯文本，本项目不涉及图片或音频生成
                    ai_parts.append(TextUIPart(text=lc_msg.content))

                    ui_messages.append(
                        UIMessage(
                            role='assistant',
                            id=lc_msg.id if lc_msg.id else uuid4(),
                            parts=ai_parts,
                        )
                    )

    return ui_messages
