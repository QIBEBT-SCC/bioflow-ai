"""简化的多模态消息构建器"""

import base64
import mimetypes
from pathlib import Path
from typing import List, Optional

import aiofiles
from langchain_core.messages import HumanMessage
from loguru import logger

from app.schemas.file import StoredFileInfo


async def build_multimodal_message(text_message: str, files: List[StoredFileInfo]) -> HumanMessage:
    """构建多模态消息

    Args:
        text_message: 用户输入的文本消息
        files: 上传的文件列表

    Returns:
        HumanMessage: 包含文本和文件内容的消息
    """
    content = []

    # 添加文本内容
    if text_message.strip():
        content.append({'type': 'text', 'text': text_message})

    # 处理文件
    for file_info in files:
        file_path = file_info.file_path
        filename = file_info.filename
        mime_type = file_info.mime_type

        # 检测文件类型
        detected_mime, _ = mimetypes.guess_type(file_path)
        file_extension = Path(file_path).suffix.lower()

        # 处理纯文本文件
        if detected_mime in ['text/plain', 'text/markdown'] or file_extension in ['.txt', '.md']:
            extracted_text = await _read_text_file(file_path)
            if extracted_text:
                content.append({'type': 'text', 'text': f'\n\n<File name="{filename}">\n{extracted_text}\n</File>'})
            continue

        # 处理图片文件
        if detected_mime and detected_mime.startswith('image/'):
            base64_data = await _encode_file_to_base64(file_path)
            if base64_data:
                content.append({'type': 'image', 'source_type': 'base64', 'data': base64_data, 'mime_type': mime_type})
            continue

        # 处理PDF等文档文件
        if detected_mime in ['application/pdf']:
            base64_data = await _encode_file_to_base64(file_path)
            if base64_data:
                content.append(
                    {
                        'type': 'file',
                        'source_type': 'base64',
                        'data': base64_data,
                        'mime_type': mime_type,
                        'filename': filename,
                    }
                )
            continue

        # 其他文件类型
        base64_data = await _encode_file_to_base64(file_path)
        if base64_data:
            content.append(
                {
                    'type': 'file',
                    'source_type': 'base64',
                    'data': base64_data,
                    'mime_type': mime_type,
                    'filename': filename,
                }
            )

    return HumanMessage(content=content)


async def _read_text_file(file_path: str) -> Optional[str]:
    """读取纯文本文件内容"""
    try:
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
            content = await file.read()
            if content.strip():
                logger.info(f'Successfully read text file, length: {len(content)} characters')
                return content
            else:
                logger.warning('Text file is empty')
                return None
    except UnicodeDecodeError:
        # 尝试其他编码
        try:
            async with aiofiles.open(file_path, 'r', encoding='gbk') as file:
                content = await file.read()
                if content.strip():
                    logger.info(f'Successfully read text file (GBK encoding), length: {len(content)} characters')
                    return content
        except Exception:
            pass

        logger.error('Cannot read text file, encoding not supported')
        return None
    except Exception as e:
        logger.error(f'Failed to read text file: {str(e)}')
        return None


async def _encode_file_to_base64(file_path: str) -> Optional[str]:
    """将文件编码为 base64 字符串"""
    try:
        async with aiofiles.open(file_path, 'rb') as f:
            content = await f.read()
            return base64.b64encode(content).decode('utf-8')
    except Exception as e:
        logger.error(f'Failed to encode file to base64: {file_path}, error: {str(e)}')
        return None
