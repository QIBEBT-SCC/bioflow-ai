def decode_redis_value(value: bytes | str | None) -> str | None:
    """Decode a Redis value returned as bytes, str, or None into str or None."""
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode()
    return value


def get_bio_db_download_task_key(db_id: int) -> str:
    """Return the Redis key for the active BioDb download task."""
    return f'bio-db:{db_id}:download-task-id'
