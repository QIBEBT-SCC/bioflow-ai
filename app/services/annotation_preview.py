"""Direct GFF and GTF parsers for genomic annotation previews."""

# Stable UI-facing parser messages are intentionally raised at their call sites.
# ruff: noqa: EM101, EM102, TRY003

import gzip
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO
from urllib.parse import unquote

ANNOTATION_KINDS = frozenset({'gff', 'gtf'})
ANNOTATION_COLUMN_COUNT = 9
GTF_ATTRIBUTE_FIELD_COUNT = 2
SEQUENCE_REGION_FIELD_COUNT = 4
CHROMOSOME_RESULT_LIMIT = 100
MAX_ANNOTATION_FEATURES = 300


class AnnotationPreviewError(ValueError):
    """A genomic annotation that cannot be previewed."""


@dataclass(frozen=True)
class AnnotationRecord:
    """One normalized 0-based, half-open annotation feature."""

    chrom: str
    start: int
    end: int
    feature_type: str
    strand: str
    name: str


def _open_annotation(path: Path) -> TextIO:
    if path.name.lower().endswith('.gz'):
        return gzip.open(path, mode='rt', encoding='utf-8', newline='')
    return path.open(encoding='utf-8', newline='')


def _gff_attributes(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    if text in {'', '.'}:
        return result
    for item in text.split(';'):
        key, separator, value = item.strip().partition('=')
        if separator and key:
            result[key] = unquote(value)
    return result


def _gtf_attributes(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    if text in {'', '.'}:
        return result
    for item in text.split(';'):
        fields = item.strip().split(maxsplit=1)
        if len(fields) != GTF_ATTRIBUTE_FIELD_COUNT:
            continue
        key, value = fields
        result[key] = value.strip().strip('"')
    return result


def _feature_name(kind: str, attributes: dict[str, str], feature_type: str) -> str:
    priorities = (
        ('Name', 'gene_name', 'transcript_name', 'gene', 'ID', 'locus_tag', 'gene_id', 'transcript_id', 'Parent')
        if kind == 'gff'
        else ('gene_name', 'transcript_name', 'exon_id', 'transcript_id', 'gene_id')
    )
    label = next((attributes[key].split(',', 1)[0] for key in priorities if attributes.get(key)), '')
    return f'{label} · {feature_type}' if label else feature_type


def _annotation_rows(path: Path, kind: str, chrom_lengths: dict[str, int]) -> Iterator[AnnotationRecord]:
    try:
        with _open_annotation(path) as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                line = raw_line.rstrip('\r\n')
                if line_number == 1:
                    line = line.lstrip('\ufeff')
                if not line:
                    continue
                if line.startswith('##FASTA'):
                    break
                if line.startswith('##sequence-region'):
                    fields = line.split()
                    if len(fields) == SEQUENCE_REGION_FIELD_COUNT:
                        try:
                            region_end = int(fields[3])
                        except ValueError as error:
                            raise AnnotationPreviewError(
                                f'Invalid sequence-region directive at line {line_number}.'
                            ) from error
                        if region_end < 1:
                            raise AnnotationPreviewError(f'Invalid sequence-region directive at line {line_number}.')
                        chrom_lengths[fields[1]] = max(chrom_lengths.get(fields[1], 0), region_end)
                    continue
                if line.startswith(('#', 'track ', 'browser ')):
                    continue
                fields = line.split('\t')
                if len(fields) != ANNOTATION_COLUMN_COUNT:
                    raise AnnotationPreviewError(
                        f'Annotation record at line {line_number} must have nine tab-separated columns.'
                    )
                chrom, _, feature_type, start_text, end_text, _, strand, _, attributes_text = fields
                try:
                    one_based_start, end = int(start_text), int(end_text)
                except ValueError as error:
                    raise AnnotationPreviewError(
                        f'Annotation coordinates are not integers at line {line_number}.'
                    ) from error
                if not chrom or not feature_type or feature_type == '.' or one_based_start < 1 or end < one_based_start:
                    raise AnnotationPreviewError(f'Invalid annotation feature at line {line_number}.')
                if strand not in {'+', '-', '.', '?'}:
                    raise AnnotationPreviewError(f'Invalid annotation strand at line {line_number}.')
                if kind == 'gff':
                    attributes = _gff_attributes(attributes_text) or _gtf_attributes(attributes_text)
                else:
                    attributes = _gtf_attributes(attributes_text)
                chrom_lengths[chrom] = max(chrom_lengths.get(chrom, 0), end)
                yield AnnotationRecord(
                    chrom=chrom,
                    start=one_based_start - 1,
                    end=end,
                    feature_type=feature_type,
                    strand=strand,
                    name=_feature_name(kind, attributes, feature_type),
                )
    except (OSError, UnicodeError) as error:
        raise AnnotationPreviewError('Could not read this UTF-8 GFF/GTF annotation.') from error


def annotation_metadata(path: Path, kind: str, query: str) -> dict:
    """Return chromosome extents for one GFF or GTF annotation."""
    if kind not in ANNOTATION_KINDS:
        raise AnnotationPreviewError('Annotation preview supports GFF and GTF files only.')
    chrom_lengths: dict[str, int] = {}
    for _ in _annotation_rows(path, kind, chrom_lengths):
        pass
    names = [name for name in chrom_lengths if query.casefold() in name.casefold()]
    return {
        'kind': kind,
        'supported': True,
        'file_size': path.stat().st_size,
        'chromosomes': [{'name': name, 'length': chrom_lengths[name]} for name in names[:CHROMOSOME_RESULT_LIMIT]],
        'has_more_chromosomes': len(names) > CHROMOSOME_RESULT_LIMIT,
        'default_chromosome': next(iter(chrom_lengths), None),
    }


def annotation_region(path: Path, kind: str, chrom: str, start: int, end: int) -> dict:
    """Return annotation features overlapping one genomic viewport."""
    if kind not in ANNOTATION_KINDS:
        raise AnnotationPreviewError('Annotation preview supports GFF and GTF files only.')
    if start < 0 or end <= start:
        raise AnnotationPreviewError('Use a non-empty, 0-based half-open interval.')
    chrom_lengths: dict[str, int] = {}
    items = []
    for record in _annotation_rows(path, kind, chrom_lengths):
        if record.chrom != chrom or record.start >= end or record.end <= start:
            continue
        items.append(
            {
                'start': record.start,
                'end': record.end,
                'name': record.name,
                'feature_type': record.feature_type,
                'strand': record.strand,
            }
        )
        if len(items) > MAX_ANNOTATION_FEATURES:
            raise AnnotationPreviewError('More than 300 annotation features overlap this region. Zoom in.')
    chrom_length = chrom_lengths.get(chrom)
    if chrom_length is None:
        raise AnnotationPreviewError('Chromosome is not present in this annotation.')
    if end > chrom_length:
        raise AnnotationPreviewError('Region extends beyond this chromosome.')
    return {'kind': 'intervals', 'chrom': chrom, 'start': start, 'end': end, 'items': items}
