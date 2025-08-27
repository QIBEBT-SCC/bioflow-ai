import hashlib
import mimetypes
from pathlib import Path
from typing import Optional, Tuple

from fastapi import UploadFile
from loguru import logger

from app.core.config import get_settings
from app.schemas.chat import ContentType
from app.schemas.file import ValidatedFileInfo


def calculate_file_hash(file_path: str, algorithm: str = 'md5') -> Optional[str]:
    """Calculate file hash value

    Args:
        file_path: File path
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')

    Returns:
        File hash value
    """
    try:
        hash_func = getattr(hashlib, algorithm)()

        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_func.update(chunk)

        return hash_func.hexdigest()

    except Exception as e:
        logger.error(f'Failed to calculate file hash: {str(e)}')
        return None


class FileValidator:
    # 危险文件扩展名黑名单
    DANGEROUS_EXTENSIONS = {
        '.exe',
        '.bat',
        '.cmd',
        '.com',
        '.scr',
        '.pif',
        '.vbs',
        '.js',
        '.jar',
        '.app',
        '.deb',
        '.dmg',
        '.pkg',
        '.rpm',
        '.msi',
        '.ps1',
        '.sh',
        '.bin',
    }

    # 文件魔术数字验证（文件头识别）
    FILE_SIGNATURES = {
        'image/jpeg': [b'\xff\xd8\xff'],
        'image/png': [b'\x89PNG\r\n\x1a\n'],
        'image/gif': [b'GIF87a', b'GIF89a'],
        'application/pdf': [b'%PDF-'],
        'application/zip': [b'PK\x03\x04', b'PK\x05\x06', b'PK\x07\x08'],
        'text/plain': [],  # 文本文件没有固定签名
    }

    def __init__(self):
        self.settings = get_settings().persistent.api_storage

    async def validate_upload_file(self, upload_file: UploadFile) -> Tuple[bool, str, Optional[ValidatedFileInfo]]:
        """验证上传文件的安全性和有效性

        Args:
            upload_file: FastAPI UploadFile对象

        Returns:
            Tuple[is_valid, error_message, validated_file_info]
        """
        try:
            # 基础验证
            if not upload_file.filename:
                raise ValueError('File name cannot be empty')

            filename = upload_file.filename
            file_extension = Path(filename).suffix.lower()

            # 文件名安全检查
            if not self._is_safe_filename(filename):
                return False, 'Filename contains illegal characters', None

            # 扩展名检查
            if file_extension in self.DANGEROUS_EXTENSIONS:
                return False, f'Uploading {file_extension} files is not allowed.', None

            # MIME类型验证
            declared_mime = upload_file.content_type or mimetypes.guess_type(filename)[0]
            if not declared_mime:
                declared_mime = 'application/octet-stream'

            content_type = self._get_content_type(declared_mime)

            is_valid, mime_error = self._check_mime_type(declared_mime, content_type)
            if not is_valid:
                return False, mime_error, None

            # 文件大小检查
            if upload_file.size is None:
                content = await upload_file.read()
                await upload_file.seek(0)
                file_size = len(content)
            else:
                file_size = upload_file.size

            is_valid, size_error = self._check_file_size(file_size, content_type)
            if not is_valid:
                return False, size_error, None

            # 文件内容验证
            content_sample = await upload_file.read(1024)
            await upload_file.seek(0)

            is_valid, content_error = self._verify_file_signature(content_sample, declared_mime)
            if not is_valid:
                return False, content_error, None

            # 创建验证文件信息对象
            validated_info = ValidatedFileInfo(
                filename=filename,
                file_extension=file_extension,
                mime_type=declared_mime,
                file_size=file_size,
                content_type=content_type,
            )

            logger.info(f'File verification passed: {filename} ({declared_mime}, {file_size} bytes)')
            return True, '', validated_info

        except Exception as e:
            logger.error(f'File verification failed: {str(e)}')
            return False, 'An error occurred during file validation', None

    @staticmethod
    def _is_safe_filename(filename: str) -> bool:
        """Check if the filename is safe"""
        # 检查路径遍历攻击
        if '..' in filename or '/' in filename or '\\' in filename:
            return False

        # 检查特殊字符
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\x00']
        if any(char in filename for char in dangerous_chars):
            return False

        # 检查长度
        if len(filename) > 255:
            return False

        return True

    def _get_content_type(self, mime_type: str) -> ContentType:
        """Determine content type based on MIME type"""
        if not mime_type:
            return ContentType.FILE

        if mime_type in self.settings.ALLOWED_IMAGE_TYPES:
            return ContentType.IMAGE
        elif mime_type in self.settings.ALLOWED_DOCUMENT_TYPES:
            return ContentType.DOCUMENT
        else:
            return ContentType.FILE

    def _check_file_size(self, file_size: int, content_type: ContentType) -> Tuple[bool, str]:
        """Check file size limit"""
        if content_type == ContentType.IMAGE:
            max_size = self.settings.MAX_IMAGE_SIZE
            max_size_mb = max_size // (1024 * 1024)
            if file_size > max_size:
                return False, f'Image file size cannot exceed {max_size_mb}MB'
        elif content_type == ContentType.DOCUMENT:
            max_size = self.settings.MAX_DOCUMENT_SIZE
            max_size_mb = max_size // (1024 * 1024)
            if file_size > max_size:
                return False, f'Document file size cannot exceed {max_size_mb}MB'
        else:
            max_size = max(self.settings.MAX_IMAGE_SIZE, self.settings.MAX_DOCUMENT_SIZE)
            max_size_mb = max_size // (1024 * 1024)
            if file_size > max_size:
                return False, f'File size cannot exceed {max_size_mb}MB'

        return True, ''

    def _check_mime_type(self, mime_type: str, content_type: ContentType) -> Tuple[bool, str]:
        """Check if MIME type is allowed"""
        if not mime_type:
            return False, 'Unable to determine file type'

        all_allowed = self.settings.ALLOWED_IMAGE_TYPES + self.settings.ALLOWED_DOCUMENT_TYPES

        if mime_type not in all_allowed:
            return False, f'Unsupported file type: {mime_type}'

        return True, ''

    def _verify_file_signature(self, content: bytes, declared_mime: str) -> Tuple[bool, str]:
        """Verify file header"""
        if not content:
            return False, 'File content is empty'

        # 对于某些类型进行严格的文件头验证
        if declared_mime in self.FILE_SIGNATURES:
            signatures = self.FILE_SIGNATURES[declared_mime]

            # 文本文件跳过签名验证
            if not signatures:
                return True, ''

            # 检查是否匹配任一签名
            for signature in signatures:
                if content.startswith(signature):
                    return True, ''

            # 对于图片文件，进行严格验证
            if declared_mime.startswith('image/'):
                return False, f'File content does not match declared type {declared_mime}'

        return True, ''


file_validator = FileValidator()
