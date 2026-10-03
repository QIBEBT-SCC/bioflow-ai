"""Generic compression and decompression executors for workflow files."""

import bz2
import gzip
import lzma
import shutil
import stat
import tarfile
import zipfile
from pathlib import Path
from typing import Literal

from bioflow_ai.schemas.archive import STREAM_COMPRESSION_FORMATS, CompressionFormat, resolve_compression_format
from bioflow_ai.schemas.react_flow import TaskInput
from bioflow_ai.schemas.workflow_config import ToolOutput
from bioflow_ai.services.workflow.execution_context import get_node_output

TarWriteMode = Literal['w', 'w:gz', 'w:bz2', 'w:xz']


def _copy_stream(source: Path, destination: Path, suffix: str) -> None:
    """Compress one regular file using a stream compression format."""
    openers = {'.gz': gzip.open, '.bz2': bz2.open, '.xz': lzma.open}
    with source.open('rb') as source_file, openers[suffix](destination, 'wb') as destination_file:
        shutil.copyfileobj(source_file, destination_file)


def _write_zip(source: Path, destination: Path) -> None:
    """Create a ZIP archive containing the source path."""
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        if source.is_dir():
            for path in source.rglob('*'):
                archive.write(path, arcname=Path(source.name) / path.relative_to(source))
        else:
            archive.write(source, arcname=source.name)


def _write_tar(source: Path, destination: Path, suffix: str) -> None:
    """Create a tar archive containing the source path."""
    modes: dict[str, TarWriteMode] = {
        '.tar': 'w',
        '.tar.gz': 'w:gz',
        '.tgz': 'w:gz',
        '.tar.bz2': 'w:bz2',
        '.tbz2': 'w:bz2',
        '.tar.xz': 'w:xz',
        '.txz': 'w:xz',
    }
    with tarfile.open(destination, modes[suffix]) as archive:
        archive.add(source, arcname=source.name)


def compress(
    *,
    inputs: dict[str, TaskInput],
    compressed_file_name: str,
    compression_format: CompressionFormat,
    output_dir: str,
) -> ToolOutput:
    """Compress an upstream path using the explicitly selected format."""
    input_path = inputs.get('input_path')
    if input_path is None:
        msg = 'compress requires exactly one input_path input'
        raise ValueError(msg)

    if (
        not compressed_file_name
        or compressed_file_name in {'.', '..'}
        or Path(compressed_file_name).name != compressed_file_name
        or '\\' in compressed_file_name
    ):
        msg = 'compressed_file_name must be a single, non-empty file name'
        raise ValueError(msg)

    resolved_format = resolve_compression_format(compressed_file_name, compression_format)

    source_path_str, sample_name = get_node_output(input_path.source_uid, input_path.source_k)
    source_path = Path(source_path_str).absolute()
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if source_path.is_dir() and resolved_format in STREAM_COMPRESSION_FORMATS:
        msg = f'{resolved_format} stream compression only supports regular files; use an archive format for directories'
        raise ValueError(msg)

    output_folder = Path(output_dir).absolute()
    output_folder.mkdir(parents=True, exist_ok=True)
    compressed_path = output_folder / compressed_file_name

    if resolved_format in STREAM_COMPRESSION_FORMATS:
        _copy_stream(source_path, compressed_path, resolved_format)
    elif resolved_format == '.zip':
        _write_zip(source_path, compressed_path)
    else:
        _write_tar(source_path, compressed_path, resolved_format)

    return ToolOutput(result={'compressed_path': compressed_path}, sample_name=sample_name)


def _safe_zip_members(archive: zipfile.ZipFile, output_folder: Path) -> None:
    """Reject ZIP members that escape the output directory or create symbolic links."""
    output_root = output_folder.resolve()
    for member in archive.infolist():
        member_path = (output_folder / member.filename).resolve()
        if not member_path.is_relative_to(output_root):
            msg = f'unsafe ZIP member path: {member.filename}'
            raise ValueError(msg)
        mode = member.external_attr >> 16
        if stat.S_ISLNK(mode):
            msg = f'ZIP symbolic links are not supported: {member.filename}'
            raise ValueError(msg)


def _decompress_stream(source: Path, destination: Path, suffix: str) -> None:
    """Decompress one stream-compressed file."""
    openers = {'.gz': gzip.open, '.bz2': bz2.open, '.xz': lzma.open}
    with openers[suffix](source, 'rb') as source_file, destination.open('wb') as destination_file:
        shutil.copyfileobj(source_file, destination_file)


def _stream_suffix_from_magic(source: Path) -> str | None:
    """Detect supported stream compression from magic bytes."""
    with source.open('rb') as file:
        magic = file.read(6)
    if magic.startswith(b'\x1f\x8b'):
        return '.gz'
    if magic.startswith(b'BZh'):
        return '.bz2'
    if magic.startswith(b'\xfd7zXZ\x00'):
        return '.xz'
    return None


def decompress(
    *,
    inputs: dict[str, TaskInput],
    output_dir: str,
) -> ToolOutput:
    """Detect and decompress a supported archive or compressed stream."""
    compressed_input = inputs.get('compressed_path')
    if compressed_input is None:
        msg = 'decompress requires exactly one compressed_path input'
        raise ValueError(msg)

    source_path_str, sample_name = get_node_output(compressed_input.source_uid, compressed_input.source_k)
    source_path = Path(source_path_str).absolute()
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if not source_path.is_file():
        raise IsADirectoryError(source_path)

    output_folder = Path(output_dir).absolute()
    output_folder.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(source_path):
        with zipfile.ZipFile(source_path) as archive:
            _safe_zip_members(archive, output_folder)
            archive.extractall(output_folder)  # noqa: S202 - members are validated above
        output_path = output_folder
    elif tarfile.is_tarfile(source_path):
        with tarfile.open(source_path) as archive:
            archive.extractall(output_folder, filter='data')
        output_path = output_folder
    elif suffix := _stream_suffix_from_magic(source_path):
        output_name = source_path.name.removesuffix(suffix)
        if not output_name:
            msg = f'cannot derive decompressed file name from {source_path.name}'
            raise ValueError(msg)
        output_path = output_folder / output_name
        _decompress_stream(source_path, output_path, suffix)
    else:
        msg = f'unsupported or unrecognized compressed file: {source_path.name}'
        raise ValueError(msg)

    return ToolOutput(result={'output_path': output_path}, sample_name=sample_name)
