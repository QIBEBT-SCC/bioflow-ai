"""Direct text parsers for bedGraph and WIG signal-track previews."""

# Stable UI-facing parser messages are intentionally raised at their call sites.
# ruff: noqa: EM101, EM102, TRY003

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

TEXT_SIGNAL_KINDS = frozenset({'bedgraph', 'wig'})
BEDGRAPH_COLUMN_COUNT = 4
VARIABLE_STEP_COLUMN_COUNT = 2
CHROMOSOME_RESULT_LIMIT = 100


class SignalPreviewError(ValueError):
    """A text signal track that cannot be previewed."""


@dataclass(frozen=True)
class SignalRecord:
    """One 0-based, half-open signal interval."""

    chrom: str
    start: int
    end: int
    value: float


def _positive_integer(value: str | None, name: str, line_number: int, *, default: int | None = None) -> int:
    if value is None and default is not None:
        return default
    try:
        result = int(value or '')
    except ValueError as error:
        raise SignalPreviewError(f'WIG {name} is not an integer at line {line_number}.') from error
    if result < 1:
        raise SignalPreviewError(f'WIG {name} must be positive at line {line_number}.')
    return result


def _signal_value(value: str, line_number: int) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise SignalPreviewError(f'Signal value is not numeric at line {line_number}.') from error
    if not math.isfinite(result):
        raise SignalPreviewError(f'Signal value must be finite at line {line_number}.')
    return result


def _directive(line: str, line_number: int) -> tuple[str, dict[str, str]]:
    fields = line.split()
    mode = fields[0]
    parameters: dict[str, str] = {}
    for field in fields[1:]:
        key, separator, value = field.partition('=')
        if not separator or not key or not value:
            raise SignalPreviewError(f'Invalid {mode} declaration at line {line_number}.')
        parameters[key] = value
    return mode, parameters


def _bedgraph_record(fields: list[str], line_number: int) -> SignalRecord:
    if len(fields) != BEDGRAPH_COLUMN_COUNT:
        raise SignalPreviewError(f'bedGraph record at line {line_number} must have four columns.')
    chrom, start_text, end_text, value_text = fields
    try:
        start, end = int(start_text), int(end_text)
    except ValueError as error:
        raise SignalPreviewError(f'bedGraph coordinates are not integers at line {line_number}.') from error
    if not chrom or start < 0 or end <= start:
        raise SignalPreviewError(f'Invalid bedGraph interval at line {line_number}.')
    return SignalRecord(chrom=chrom, start=start, end=end, value=_signal_value(value_text, line_number))


def _bedgraph_rows(path: Path) -> Iterator[SignalRecord]:
    with path.open(encoding='utf-8') as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line or line.startswith(('#', 'track ', 'browser ')):
                continue
            yield _bedgraph_record(line.split(), line_number)


def _wig_rows(path: Path) -> Iterator[SignalRecord]:
    mode = ''
    chrom = ''
    span = 1
    step = 1
    next_start = 0
    with path.open(encoding='utf-8') as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line or line.startswith(('#', 'track ', 'browser ')):
                continue
            if line.split(maxsplit=1)[0] in {'fixedStep', 'variableStep'}:
                mode, parameters = _directive(line, line_number)
                chrom = parameters.get('chrom', '')
                if not chrom:
                    raise SignalPreviewError(f'WIG declaration is missing chrom at line {line_number}.')
                span = _positive_integer(parameters.get('span'), 'span', line_number, default=1)
                if mode == 'fixedStep':
                    start = _positive_integer(parameters.get('start'), 'start', line_number)
                    step = _positive_integer(parameters.get('step'), 'step', line_number, default=1)
                    next_start = start - 1
                continue
            fields = line.split()
            if not mode:
                raise SignalPreviewError(
                    f'WIG data appears before a fixedStep or variableStep declaration at line {line_number}.'
                )
            if mode == 'fixedStep':
                if len(fields) != 1:
                    raise SignalPreviewError(f'fixedStep data at line {line_number} must contain one value.')
                record = SignalRecord(
                    chrom=chrom,
                    start=next_start,
                    end=next_start + span,
                    value=_signal_value(fields[0], line_number),
                )
                next_start += step
                yield record
                continue
            if len(fields) != VARIABLE_STEP_COLUMN_COUNT:
                raise SignalPreviewError(f'variableStep data at line {line_number} must contain position and value.')
            position = _positive_integer(fields[0], 'position', line_number)
            start = position - 1
            yield SignalRecord(
                chrom=chrom,
                start=start,
                end=start + span,
                value=_signal_value(fields[1], line_number),
            )


def _signal_rows(path: Path, kind: str) -> Iterator[SignalRecord]:
    if kind == 'bedgraph':
        yield from _bedgraph_rows(path)
        return
    if kind == 'wig':
        yield from _wig_rows(path)
        return
    raise SignalPreviewError('Text signal preview supports bedGraph and WIG files only.')


def signal_metadata(path: Path, kind: str, query: str) -> dict:
    """Return chromosome extents for one text signal track."""
    chroms: dict[str, int] = {}
    try:
        for record in _signal_rows(path, kind):
            chroms[record.chrom] = max(chroms.get(record.chrom, 0), record.end)
    except (OSError, UnicodeError) as error:
        raise SignalPreviewError('Could not read this UTF-8 signal track.') from error
    names = [name for name in chroms if query.casefold() in name.casefold()]
    return {
        'kind': kind,
        'supported': True,
        'file_size': path.stat().st_size,
        'chromosomes': [{'name': name, 'length': chroms[name]} for name in names[:CHROMOSOME_RESULT_LIMIT]],
        'has_more_chromosomes': len(names) > CHROMOSOME_RESULT_LIMIT,
        'default_chromosome': next(iter(chroms), None),
    }


def signal_region(path: Path, kind: str, chrom: str, start: int, end: int, bins: int) -> dict:
    """Return overlap-weighted mean signal values for a genomic viewport."""
    if start < 0 or end <= start:
        raise SignalPreviewError('Use a non-empty, 0-based half-open interval.')
    bin_count = min(bins, end - start)
    span = end - start
    sums = [0.0] * bin_count
    weights = [0.0] * bin_count
    chrom_length = 0
    try:
        for record in _signal_rows(path, kind):
            if record.chrom != chrom:
                continue
            chrom_length = max(chrom_length, record.end)
            overlap_start = max(start, record.start)
            overlap_end = min(end, record.end)
            if overlap_start >= overlap_end:
                continue
            first_bin = min(bin_count - 1, (overlap_start - start) * bin_count // span)
            last_bin = min(bin_count - 1, (overlap_end - 1 - start) * bin_count // span)
            for index in range(first_bin, last_bin + 1):
                bin_start = start + index * span / bin_count
                bin_end = start + (index + 1) * span / bin_count
                overlap = max(0.0, min(overlap_end, bin_end) - max(overlap_start, bin_start))
                sums[index] += record.value * overlap
                weights[index] += overlap
    except (OSError, UnicodeError) as error:
        raise SignalPreviewError('Could not read this UTF-8 signal track.') from error
    if not chrom_length:
        raise SignalPreviewError('Chromosome is not present in this file.')
    if end > chrom_length:
        raise SignalPreviewError('Region extends beyond this chromosome.')
    return {
        'kind': 'signal',
        'chrom': chrom,
        'start': start,
        'end': end,
        'points': [total / weight if weight else None for total, weight in zip(sums, weights, strict=True)],
    }
