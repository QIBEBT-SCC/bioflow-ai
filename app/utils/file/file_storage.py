import mimetypes
from pathlib import Path
from typing import Optional
from datetime import datetime, UTC
import base64

import aiofiles
from fastapi import UploadFile, HTTPException
from loguru import logger
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR
from uuid_extensions import uuid7

from app.core.config import get_settings
from app.schemas.file import ValidatedFileInfo, StoredFileInfo


class FileStorageService:
    def __init__(self, storage_base_path: Optional[str] = None):
        self.settings = get_settings()
        self.storage_base = Path(storage_base_path or self.settings.persistent.api_storage.UPLOAD_DIR)

    async def save_file(
        self, upload_file: UploadFile, user_name: str, validated_info: ValidatedFileInfo
    ) -> StoredFileInfo:
        """Save the verified file

        Args:
            upload_file: FastAPI UploadFile object
            user_name: User name
            validated_info: Verified File Information

        Returns:
            StoredFileInfo: Store file information object
        """

        file_id = str(uuid7())
        date_path = datetime.now(UTC).strftime('%Y-%m-%d')
        target_dir = self.storage_base / user_name / date_path
        target_dir.mkdir(parents=True, exist_ok=True)

        file_path = target_dir / f'{file_id}{validated_info.file_extension}'

        try:
            async with aiofiles.open(file_path, 'wb') as f:
                content = await upload_file.read()
                await f.write(content)

            logger.info(f'File save successfully: {file_path}')
            stored_info = StoredFileInfo(file_id=file_id, file_path=str(file_path), original_info=validated_info)

            return stored_info

        except Exception as e:
            logger.error(f'File save failed: {str(e)}')
            if file_path.exists():
                file_path.unlink()
            raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail='File save failed')


file_storage = FileStorageService()


async def encode_image_to_base64(file_path: str) -> str:
    """Encode image file to base64 string"""
    try:
        async with aiofiles.open(file_path, 'rb') as f:
            content = await f.read()
            return base64.b64encode(content).decode('utf-8')
    except Exception as e:
        logger.error(f'Image base64 encoding failed: {str(e)}')
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail='Image encoding failed')


def get_image_data_url(file_path: str, base64_content: str) -> str:
    """生成图片data URL"""
    mime_type = mimetypes.guess_type(file_path)[0] or 'image/jpeg'
    return f'data:{mime_type};base64,{base64_content}'
