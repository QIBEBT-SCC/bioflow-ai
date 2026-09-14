"""Short-lived, resource-limited parser process for run-file previews."""

# Stable machine-readable error codes are intentionally passed as string literals.
# ruff: noqa: EM101, PLR2004

import json
import math
import resource
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

MAX_BED_BYTES = 4 * 1024 * 1024
MAX_BED_RECORDS = 50_000
MAX_INTERVALS = 300
MAX_POINTS = 512
MAX_RESPONSE_BYTES = 256 * 1024


class PreviewError(Exception):
    """Expected failure with a stable UI-facing code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def file_kind(path: Path) -> str:
    """Choose a parser without sniffing arbitrary binary content as text."""
    suffix = path.suffix.lower()
    if suffix in {'.bw', '.bigwig'}:
        return 'bigwig'
    if suffix in {'.bb', '.bigbed'}:
        return 'bigbed'
    if suffix == '.bed':
        return 'bed'
    raise PreviewError('unsupported', 'This file format has no genomic preview.')


def _bed_rows(path: Path) -> Iterator[tuple[str, int, int, str]]:
    if path.stat().st_size > MAX_BED_BYTES:
        raise PreviewError('too_large', 'BED preview supports files up to 4 MiB.')
    count = 0
    try:
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                if not line.strip() or line.startswith(('#', 'track ', 'browser ')):
                    continue
                count += 1
                if count > MAX_BED_RECORDS:
                    raise PreviewError('too_large', 'BED preview supports up to 50,000 records.')
                fields = line.rstrip('\r\n').split('\t')
                if len(fields) < 3:
                    raise PreviewError('invalid', f'Invalid BED record at line {count}.')
                chrom, start_text, end_text = fields[:3]
                start, end = int(start_text), int(end_text)
                if not chrom or start < 0 or end <= start:
                    raise PreviewError('invalid', f'Invalid BED coordinates at record {count}.')
                yield chrom, start, end, fields[3][:100] if len(fields) > 3 else ''
    except (UnicodeError, ValueError) as error:
        raise PreviewError('invalid', 'BED file is not valid UTF-8 BED data.') from error


def _chromosomes(chroms: dict[str, int], query: str) -> dict[str, Any]:
    names = [name for name in chroms if query.lower() in name.lower()]
    return {
        'chromosomes': [{'name': name, 'length': chroms[name]} for name in names[:100]],
        'has_more_chromosomes': len(names) > 100,
        'default_chromosome': next(iter(chroms), None),
    }


def _metadata(path: Path, kind: str, query: str) -> dict[str, Any]:
    if kind == 'bed':
        chroms: dict[str, int] = {}
        for chrom, _, end, _ in _bed_rows(path):
            chroms[chrom] = max(chroms.get(chrom, 0), end)
    else:
        import pyBigWig

        try:
            with pyBigWig.open(str(path)) as big_file:
                if not big_file or bool(big_file.isBigWig()) != (kind == 'bigwig'):
                    raise PreviewError('invalid', 'File contents do not match its bigWig/bigBed extension.')
                chroms = big_file.chroms()
        except RuntimeError as error:
            raise PreviewError('invalid', 'The indexed genomic file is damaged or incomplete.') from error
    return {'kind': kind, 'supported': True, 'file_size': path.stat().st_size, **_chromosomes(chroms, query)}


def _region(path: Path, kind: str, chrom: str, start: int, end: int, bins: int) -> dict[str, Any]:
    if start < 0 or end <= start:
        raise PreviewError('invalid_region', 'Use a non-empty, 0-based half-open interval.')
    max_span = 10_000_000 if kind == 'bigwig' else 1_000_000
    if end - start > max_span:
        raise PreviewError('range_too_large', f'This track allows a region of at most {max_span:,} bases.')
    if kind == 'bed':
        items = []
        chrom_length = 0
        for row_chrom, row_start, row_end, name in _bed_rows(path):
            if row_chrom != chrom:
                continue
            chrom_length = max(chrom_length, row_end)
            if row_start < end and row_end > start:
                items.append({'start': row_start, 'end': row_end, 'name': name})
                if len(items) > MAX_INTERVALS:
                    raise PreviewError('too_dense', 'More than 300 features overlap this region. Zoom in.')
        if not chrom_length:
            raise PreviewError('invalid_region', 'Chromosome is not present in this file.')
        if end > chrom_length:
            raise PreviewError('invalid_region', 'Region extends beyond this chromosome.')
        return {'kind': 'intervals', 'chrom': chrom, 'start': start, 'end': end, 'items': items}

    import pyBigWig

    try:
        with pyBigWig.open(str(path)) as big_file:
            if not big_file or bool(big_file.isBigWig()) != (kind == 'bigwig'):
                raise PreviewError('invalid', 'File contents do not match its bigWig/bigBed extension.')
            chrom_length = big_file.chroms(chrom)
            if chrom_length is None or end > chrom_length:
                raise PreviewError('invalid_region', 'Region is outside this chromosome.')
            if kind == 'bigwig':
                values = big_file.stats(chrom, start, end, nBins=min(bins, end - start), type='mean')
                points = [value if value is not None and math.isfinite(value) else None for value in values]
                return {'kind': 'signal', 'chrom': chrom, 'start': start, 'end': end, 'points': points}
            entries = big_file.entries(chrom, start, end) or []
            if len(entries) > MAX_INTERVALS:
                raise PreviewError('too_dense', 'More than 300 features overlap this region. Zoom in.')
            items = [
                {'start': row_start, 'end': row_end, 'name': description.split('\t', 1)[0][:100]}
                for row_start, row_end, description in entries
            ]
            return {'kind': 'intervals', 'chrom': chrom, 'start': start, 'end': end, 'items': items}
    except RuntimeError as error:
        raise PreviewError('invalid', 'The indexed genomic file is damaged or incomplete.') from error


def parse(request: dict[str, Any]) -> dict[str, Any]:
    """Validate file identity inside the worker, then parse one bounded request."""
    root = Path(request['root']).resolve()
    path = Path(request['path']).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise PreviewError('missing', 'Run file is missing or outside the run directory.')
    initial = path.stat()
    if initial.st_size != request['size'] or initial.st_mtime_ns != request['mtime_ns']:
        raise PreviewError('changing', 'File changed during preview. Refresh and try again.')
    kind = file_kind(path)
    if request['operation'] == 'meta':
        result = _metadata(path, kind, request.get('query', ''))
    else:
        result = _region(path, kind, request['chrom'], request['start'], request['end'], request['bins'])
    final = path.stat()
    if final.st_size != initial.st_size or final.st_mtime_ns != initial.st_mtime_ns:
        raise PreviewError('changing', 'File changed during preview. Refresh and try again.')
    return result


def main() -> None:
    """Serve exactly one request and exit so native parsers cannot linger."""
    # Native BBI parsing is isolated so timeout and memory limits are enforceable.
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    try:
        result = {'ok': True, 'data': parse(json.loads(sys.stdin.buffer.read(MAX_RESPONSE_BYTES)))}
        encoded = json.dumps(result, ensure_ascii=False, allow_nan=False).encode()
        if len(encoded) > MAX_RESPONSE_BYTES:
            encoded = json.dumps(
                {
                    'ok': False,
                    'code': 'too_large',
                    'message': 'Preview response exceeded 256 KiB.',
                }
            ).encode()
    except PreviewError as error:
        encoded = json.dumps({'ok': False, 'code': error.code, 'message': str(error)}).encode()
    except (OSError, MemoryError, TypeError, ValueError) as error:
        encoded = json.dumps(
            {'ok': False, 'code': 'invalid', 'message': f'File preview failed: {type(error).__name__}.'}
        ).encode()
    sys.stdout.buffer.write(encoded)


if __name__ == '__main__':
    main()
