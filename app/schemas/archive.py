"""Compression format definitions shared by workflow schemas and tasks."""

from typing import Literal

CompressionFormat = Literal[
    '.zip',
    '.tar.gz',
    '.tar.bz2',
    '.tar.xz',
    '.tar',
    '.gz',
    '.bz2',
    '.xz',
]

COMPRESSION_FORMAT_BY_SUFFIX: dict[str, CompressionFormat] = {
    '.zip': '.zip',
    '.tar.gz': '.tar.gz',
    '.tar.bz2': '.tar.bz2',
    '.tar.xz': '.tar.xz',
    '.tar': '.tar',
    '.gz': '.gz',
    '.bz2': '.bz2',
    '.xz': '.xz',
}
SUPPORTED_COMPRESSION_SUFFIXES = frozenset(COMPRESSION_FORMAT_BY_SUFFIX)
SUPPORTED_COMPRESSION_FORMATS = frozenset(COMPRESSION_FORMAT_BY_SUFFIX.values())
STREAM_COMPRESSION_FORMATS = frozenset({'.gz', '.bz2', '.xz'})


def get_compression_suffix(file_name: str) -> str | None:
    """Return the longest supported compression suffix for a file name."""
    return next(
        (
            suffix
            for suffix in sorted(SUPPORTED_COMPRESSION_SUFFIXES, key=len, reverse=True)
            if file_name.endswith(suffix)
        ),
        None,
    )


def resolve_compression_format(
    file_name: str,
    compression_format: str,
) -> CompressionFormat:
    """Validate an explicit format and its corresponding file name suffix."""
    suffix = get_compression_suffix(file_name)
    if suffix is None:
        supported = ', '.join(sorted(SUPPORTED_COMPRESSION_SUFFIXES))
        msg = f'unsupported compression suffix; supported suffixes: {supported}'
        raise ValueError(msg)

    if compression_format not in SUPPORTED_COMPRESSION_FORMATS:
        supported = ', '.join(sorted(SUPPORTED_COMPRESSION_FORMATS))
        msg = f'unsupported compression format; supported formats: {supported}'
        raise ValueError(msg)
    inferred_format = COMPRESSION_FORMAT_BY_SUFFIX[suffix]
    if compression_format != inferred_format:
        msg = f'compressed_file_name suffix {suffix} does not match compression_format {compression_format}'
        raise ValueError(msg)
    return inferred_format
