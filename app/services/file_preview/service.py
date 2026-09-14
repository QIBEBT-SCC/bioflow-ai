"""Run one preview request in a disposable parser process."""

import asyncio
import json
import os
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from app.services.file_preview.worker import MAX_RESPONSE_BYTES

PREVIEW_TIMEOUT_SECONDS = 2
_slots: asyncio.Queue[None] = asyncio.Queue(maxsize=2)
_slots.put_nowait(None)
_slots.put_nowait(None)
_ERROR_STATUS = {
    'unsupported': 415,
    'missing': 404,
    'changing': 409,
    'too_large': 413,
    'range_too_large': 413,
    'too_dense': 422,
    'invalid_region': 422,
    'invalid': 422,
}


async def run_preview(root: Path, path: Path, operation: str, **kwargs: Any) -> dict[str, Any]:
    """Return quickly under load and kill a parser that exceeds two seconds."""
    try:
        _slots.get_nowait()
    except asyncio.QueueEmpty as error:
        raise HTTPException(429, 'Preview is busy. Please retry.') from error
    process = None
    try:
        try:
            async with asyncio.timeout(PREVIEW_TIMEOUT_SECONDS):
                stat = await asyncio.to_thread(path.stat)
                request = {
                    'root': str(root),
                    'path': str(path),
                    'size': stat.st_size,
                    'mtime_ns': stat.st_mtime_ns,
                    'operation': operation,
                    **kwargs,
                }
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    '-m',
                    'app.services.file_preview.worker',
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                    env={
                        **os.environ,
                        'OPENBLAS_NUM_THREADS': '1',
                        'OMP_NUM_THREADS': '1',
                        'MKL_NUM_THREADS': '1',
                        'NUMEXPR_NUM_THREADS': '1',
                    },
                )
                stdout, _ = await process.communicate(json.dumps(request).encode())
        except TimeoutError as error:
            raise HTTPException(504, 'Preview exceeded the two-second time limit.') from error
        if process.returncode or len(stdout) > MAX_RESPONSE_BYTES:
            raise HTTPException(422, 'Preview could not read this file within resource limits.')
        response = json.loads(stdout)
        if not response['ok']:
            raise HTTPException(_ERROR_STATUS.get(response['code'], 422), response['message'])
        return response['data']
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise HTTPException(422, 'Preview parser failed to read this file.') from error
    finally:
        if process is not None and process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
            await asyncio.shield(process.wait())
        _slots.put_nowait(None)
