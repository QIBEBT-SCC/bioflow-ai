"""Object storage utility functions for handling file uploads and downloads"""

import base64
import io
from pathlib import Path
from typing import Optional

from botocore.exceptions import ClientError
from loguru import logger

from app.core.config import get_settings
from app.core.s3_client import s3_client


class ObjectStorageError(Exception):
    """Exception for object storage related errors"""

    pass


def _get_bucket():
    """Get S3 bucket object"""
    bucket_name = get_settings().persistent.storage.s3.BUCKET
    return s3_client.Bucket(bucket_name)


def generate_key_from_hash(file_hash: str, filename: str, prefix: str = '') -> str:
    """
    Generate object storage key based on file hash

    Args:
        file_hash: SHA256 hash of the file
        filename: Original filename
        prefix: Key prefix for organizing file structure

    Returns:
        Generated key in format: prefix/hash[:2]/hash[2:4]/hash[:16]_filename

    Example:
        >>> key = generate_key_from_hash('a1b2c3d4e5f6...', 'report.pdf', prefix='documents')
        >>> # Returns: "documents/a1/b2/c3/report.pdf"
    """
    # Use first 2 chars and next 2 chars for two-level directory sharding
    shard1 = file_hash[:2]
    shard2 = file_hash[2:4]
    shard3 = file_hash[4:6]

    # Build key: prefix/shard1/shard2/filename
    if prefix:
        key = f'{prefix}/{shard1}/{shard2}/{shard3}/{filename}'
    else:
        key = f'{shard1}/{shard2}/{shard3}/{filename}'

    return key


def upload_file(file_path: str | Path, file_hash: str, prefix: str = '') -> str:
    """
    Upload file to object storage

    Args:
        file_path: Path to the file to upload
        file_hash: SHA256 hash of the file
        prefix: Key prefix for organizing file structure

    Returns:
        Uploaded file key

    Raises:
        ObjectStorageError: Raised when upload fails

    Example:
        >>> key = upload_file('/path/to/file.pdf', prefix='documents')
        >>> # Returns: "documents/a1/b2/c3/report.pdf"
    """
    try:
        bucket = _get_bucket()
        file_path = Path(file_path) if isinstance(file_path, str) else file_path

        # Check file exists
        if not file_path.exists():
            raise ObjectStorageError(f'File does not exist: {file_path}')

        key = generate_key_from_hash(file_hash, file_path.name, prefix=prefix)

        logger.debug(f'Uploading file: {file_path} -> {key}')
        bucket.upload_file(str(file_path), key)
        logger.success(f'File uploaded successfully: {key}')

        return key

    except ClientError as e:
        error_msg = f'Failed to upload file: {e}'
        logger.error(error_msg)
        raise ObjectStorageError(error_msg) from e
    except Exception as e:
        error_msg = f'Error occurred while uploading file: {e}'
        logger.error(error_msg)
        raise ObjectStorageError(error_msg) from e


def download_file(key: str, destination: Optional[str | Path] = None) -> Optional[Path]:
    """
    Download file from object storage

    Args:
        key: Object storage key
        destination: Download destination path, uses temp directory if not provided

    Returns:
        downloaded file path

    Raises:
        ObjectStorageError: Raised when download fails

    Example:
        >>> # Download to specific path
        >>> file_path = download_file('documents/20231225/report.pdf', destination='/tmp/report.pdf')
    """
    try:
        bucket = _get_bucket()
        obj = bucket.Object(key)

        # Download to file
        if destination:
            dest_path = Path(destination).expanduser().resolve()
        else:
            # Use temp directory
            temp_dir = Path(get_settings().persistent.storage.STORAGE_PATH).expanduser().resolve() / 'temp'
            temp_dir.mkdir(parents=True, exist_ok=True)

            filename = Path(key).name
            dest_path = temp_dir / filename

        # Ensure destination directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        logger.debug(f'Downloading file: {key} -> {dest_path}')
        obj.download_file(str(dest_path))
        logger.success(f'File downloaded successfully: {dest_path}')
        return dest_path

    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        if error_code == '404' or error_code == 'NoSuchKey':
            error_msg = f'File does not exist: {key}'
        else:
            error_msg = f'Failed to download file: {e}'
        logger.error(error_msg)
        raise ObjectStorageError(error_msg) from e
    except Exception as e:
        error_msg = f'Error occurred while downloading file: {e}'
        logger.error(error_msg)
        raise ObjectStorageError(error_msg) from e


def upload_file_from_base64(base64_str: str, filename: str, file_hash: str, prefix: str = '') -> str:
    """
    Upload file from base64 encoded string to object storage

    Args:
        base64_str: Base64 encoded file data string
        filename: Original filename (required for key generation)
        file_hash: SHA256 hash of the file
        prefix: Key prefix for organizing file structure

    Returns:
        Uploaded file key

    Raises:
        ObjectStorageError: Raised when upload fails

    Example:
        >>> base64_image = 'iVBORw0KGgoAAAANSUhEUgAAAAUA...'
        >>> key = upload_file_from_base64(base64_image, filename='avatar.png', file_hash='xxxxxxx', prefix='avatars')
        >>> # Returns: "avatars/3f/8a/64/avatar.png"
    """
    try:
        bucket = _get_bucket()

        # Decode base64 string
        try:
            file_data = base64.b64decode(base64_str)
        except Exception as e:
            raise ObjectStorageError(f'Failed to decode base64: {e}')

        key = generate_key_from_hash(file_hash, filename, prefix=prefix)

        logger.debug(f'Uploading file from base64 data (size: {len(file_data)} bytes) -> {key}')
        file_obj = io.BytesIO(file_data)
        bucket.upload_fileobj(file_obj, key)
        logger.success(f'File uploaded successfully from base64: {key}')

        return key

    except ClientError as e:
        error_msg = f'Failed to upload file from base64: {e}'
        logger.error(error_msg)
        raise ObjectStorageError(error_msg) from e
    except Exception as e:
        error_msg = f'Error occurred while uploading file from base64: {e}'
        logger.error(error_msg)
        raise ObjectStorageError(error_msg) from e


def download_file_as_base64(key: str) -> str:
    """
    Download file from object storage and return as base64 encoded string

    Args:
        key: Object storage key

    Returns:
        Base64 encoded file data string

    Raises:
        ObjectStorageError: Raised when download fails

    Example:
        >>> base64_str = download_file_as_base64('avatars/3f/8a/64/avatar.png')
        >>> # Send base64_str to frontend or API
    """
    try:
        bucket = _get_bucket()
        obj = bucket.Object(key)

        # Download file to memory
        logger.debug(f'Downloading file to memory: {key}')
        buffer = io.BytesIO()
        obj.download_fileobj(buffer)
        buffer.seek(0)
        file_bytes = buffer.read()

        # Encode as base64
        base64_str = base64.b64encode(file_bytes).decode('utf-8')

        logger.success(
            f'File downloaded and encoded to base64 successfully: {key} ({len(file_bytes)} bytes -> {len(base64_str)} chars)'
        )
        return base64_str

    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        if error_code == '404' or error_code == 'NoSuchKey':
            error_msg = f'File does not exist: {key}'
        else:
            error_msg = f'Failed to download file as base64: {e}'
        logger.error(error_msg)
        raise ObjectStorageError(error_msg) from e
    except Exception as e:
        error_msg = f'Error occurred while downloading file as base64: {e}'
        logger.error(error_msg)
        raise ObjectStorageError(error_msg) from e


def delete_file(key: str) -> bool:
    """
    Delete file

    Args:
        key: Object storage key

    Returns:
        Returns True if deletion succeeds

    Raises:
        ObjectStorageError: Raised when deletion fails

    Example:
        >>> delete_file('documents/20231225/old_report.pdf')
        True
    """
    try:
        bucket = _get_bucket()
        logger.info(f'Deleting file: {key}')
        obj = bucket.Object(key)
        obj.delete()
        logger.success(f'File deleted successfully: {key}')
        return True

    except ClientError as e:
        error_msg = f'Failed to delete file: {e}'
        logger.error(error_msg)
        raise ObjectStorageError(error_msg) from e


def file_exists(key: str) -> bool:
    """
    Check if file exists

    Args:
        key: Object storage key

    Returns:
        Returns True if file exists, False otherwise

    Example:
        >>> if file_exists('documents/20231225/report.pdf'):
        ...     print('File exists')
    """
    try:
        bucket = _get_bucket()
        obj = bucket.Object(key)
        obj.load()
        return True
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        if error_code == '404' or error_code == 'NoSuchKey':
            return False
        logger.error(f'Failed to check file existence: {e}')
        return False


def get_presigned_url(key: str, expires_in: int = None) -> str:
    """
    Generate presigned URL for temporary file access

    Args:
        key: Object storage key
        expires_in: URL expiration time in seconds, uses config default if not provided

    Returns:
        Presigned URL

    Raises:
        ObjectStorageError: Raised when generation fails

    Example:
        >>> # Generate URL valid for 1 hour
        >>> url = get_presigned_url('documents/20231225/report.pdf', expires_in=3600)
        >>> # Use URL to share file temporarily
    """
    try:
        settings = get_settings().persistent.storage.s3
        bucket_name = settings.BUCKET

        if expires_in is None:
            expires_in = settings.PRESIGN_EXPIRE

        url = s3_client.meta.client.generate_presigned_url(
            'get_object', Params={'Bucket': bucket_name, 'Key': key}, ExpiresIn=expires_in
        )

        logger.info(f'Generated presigned URL: {key} (expires in: {expires_in}s)')
        return url

    except ClientError as e:
        error_msg = f'Failed to generate presigned URL: {e}'
        logger.error(error_msg)
        raise ObjectStorageError(error_msg) from e
