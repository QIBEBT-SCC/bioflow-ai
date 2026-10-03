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
CONTACT_KINDS = frozenset({'hic', 'cool', 'mcool'})
MAX_CONTACT_BINS = 64
MAX_CONTACT_SPAN = 10_000_000


class PreviewError(Exception):
    """Expected failure with a stable UI-facing code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def file_kind(path: Path) -> str:
    """Choose a parser without sniffing arbitrary binary content as text."""
    name = path.name.lower()
    if name.endswith(('.gff.gz', '.gff3.gz')):
        return 'gff'
    if name.endswith(('.gtf.gz', '.gtt.gz')):
        return 'gtf'
    suffix = path.suffix.lower()
    if suffix in {'.bw', '.bigwig'}:
        return 'bigwig'
    if suffix in {'.bb', '.bigbed'}:
        return 'bigbed'
    if suffix == '.bed':
        return 'bed'
    if suffix in {'.bedgraph', '.bdg', '.bg'}:
        return 'bedgraph'
    if suffix in {'.wig', '.wiggle'}:
        return 'wig'
    if suffix in {'.gff', '.gff3'}:
        return 'gff'
    if suffix in {'.gtf', '.gtt'}:
        return 'gtf'
    if suffix in {'.hic', '.cool', '.mcool'}:
        return suffix[1:]
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
    if kind in CONTACT_KINDS:
        return _contact_metadata(path, kind, query)
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


def _contact_file(path: Path, kind: str) -> tuple[dict[str, int], list[int]]:
    import hictkpy

    try:
        matches = {
            'hic': hictkpy.is_hic,
            'cool': hictkpy.is_cooler,
            'mcool': hictkpy.is_mcool_file,
        }
        if not matches[kind](str(path)):
            raise PreviewError('invalid', 'File contents do not match its Hi-C extension.')
        if kind == 'cool':
            file = hictkpy.File(str(path))
            return dict(file.chromosomes()), [int(file.resolution())]
        file = hictkpy.MultiResFile(str(path))
        return dict(file.chromosomes()), sorted({int(resolution) for resolution in file.resolutions()})
    except RuntimeError as error:
        raise PreviewError('invalid', 'The indexed Hi-C file is damaged or incomplete.') from error


def _contact_metadata(path: Path, kind: str, query: str) -> dict[str, Any]:
    chroms, resolutions = _contact_file(path, kind)
    if not resolutions or any(resolution <= 0 for resolution in resolutions):
        raise PreviewError('invalid', 'The Hi-C file has no valid resolution.')
    return {
        'kind': kind,
        'supported': True,
        'file_size': path.stat().st_size,
        'resolutions': resolutions[:100],
        **_chromosomes(chroms, query),
    }


def _contact_region(path: Path, kind: str, request: dict[str, Any]) -> dict[str, Any]:
    import hictkpy

    chroms, resolutions = _contact_file(path, kind)
    chrom1, start1, end1 = request['chrom'], request['start'], request['end']
    chrom2 = request.get('chrom2') or chrom1
    start2 = request.get('start2')
    end2 = request.get('end2')
    if (start2 is None) != (end2 is None):
        raise PreviewError('invalid_region', 'Both coordinates of the second locus are required.')
    if start2 is None:
        start2, end2 = start1, end1
    axes = ((chrom1, start1, end1), (chrom2, start2, end2))
    for chrom, start, end in axes:
        if chrom not in chroms or start < 0 or end <= start or end > chroms[chrom]:
            raise PreviewError('invalid_region', 'Region is outside this chromosome.')
        if end - start > MAX_CONTACT_SPAN:
            raise PreviewError('range_too_large', 'Hi-C viewports allow at most 10 Mb per axis.')

    def bin_count(start: int, end: int, resolution: int) -> int:
        return (end - 1) // resolution - start // resolution + 1

    selected = request.get('resolution')
    if selected is not None and selected not in resolutions:
        raise PreviewError('invalid_region', 'Resolution is not present in this Hi-C file.')
    candidates = [selected] if selected is not None else resolutions
    resolution = next(
        (
            value
            for value in candidates
            if bin_count(start1, end1, value) <= MAX_CONTACT_BINS and bin_count(start2, end2, value) <= MAX_CONTACT_BINS
        ),
        None,
    )
    if resolution is None:
        raise PreviewError(
            'too_dense', 'This resolution exceeds 64 bins per axis. Zoom in or choose a coarser resolution.'
        )
    width = bin_count(start1, end1, resolution)
    height = bin_count(start2, end2, resolution)
    values = [0.0] * (width * height)
    first_bin1, first_bin2 = start1 // resolution, start2 // resolution
    try:
        file = hictkpy.File(str(path), resolution)
        first = f'{chrom1}:{start1 + 1}-{end1}'
        second = f'{chrom2}:{start2 + 1}-{end2}'
        chrom_order = {name: index for index, name in enumerate(chroms)}
        swapped = (chrom_order[chrom1], start1) > (chrom_order[chrom2], start2)
        for pixel in file.fetch(
            second if swapped else first,
            first if swapped else second,
            join=True,
            count_type='float64',
        ):
            first_start = pixel.start2 if swapped else pixel.start1
            second_start = pixel.start1 if swapped else pixel.start2
            x = first_start // resolution - first_bin1
            y = second_start // resolution - first_bin2
            if not (0 <= x < width and 0 <= y < height):
                continue
            count = float(pixel.count)
            if not math.isfinite(count):
                continue
            values[y * width + x] += count
            if chrom1 == chrom2:
                mirror_x = second_start // resolution - first_bin1
                mirror_y = first_start // resolution - first_bin2
                if (mirror_x, mirror_y) != (x, y) and 0 <= mirror_x < width and 0 <= mirror_y < height:
                    values[mirror_y * width + mirror_x] += count
    except RuntimeError as error:
        raise PreviewError('invalid', 'The indexed Hi-C file is damaged or incomplete.') from error
    return {
        'kind': 'heatmap',
        'chrom': chrom1,
        'start': start1,
        'end': end1,
        'chrom2': chrom2,
        'start2': start2,
        'end2': end2,
        'resolution': resolution,
        'width': width,
        'height': height,
        'values': values,
    }


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
    elif kind in CONTACT_KINDS:
        result = _contact_region(path, kind, request)
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
