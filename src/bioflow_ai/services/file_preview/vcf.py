"""Direct parser and query helpers for text VCF previews."""

# Stable UI-facing parser messages are intentionally raised at their call sites.
# ruff: noqa: EM101, EM102, TRY003, TRY301

import gzip
import math
import re
from contextlib import suppress
from pathlib import Path
from typing import TextIO

from pydantic import BaseModel, ConfigDict, Field

QUOTED_VALUE_MIN_LENGTH = 2
REQUIRED_COLUMN_COUNT = 8
FORMAT_COLUMN_INDEX = 8
SAMPLE_COLUMN_START = 9


class VcfPreviewError(ValueError):
    """A VCF file or preview query that cannot be evaluated."""


class VcfRecord(BaseModel):
    """One parsed VCF body record."""

    model_config = ConfigDict(frozen=True)

    index: int
    line_number: int
    chrom: str
    pos: int
    identifier: str
    ref: str
    alts: tuple[str, ...]
    qual: float | None
    filter_text: str
    info_text: str
    format_keys: tuple[str, ...]
    sample_values: tuple[str, ...]


class VcfDocument(BaseModel):
    """VCF header metadata and records loaded from one file."""

    version: str
    samples: list[str]
    records: list[VcfRecord]
    contig_order: list[str] = Field(default_factory=list)
    contig_lengths: dict[str, int] = Field(default_factory=dict)
    info_definitions: dict[str, dict[str, str]] = Field(default_factory=dict)
    format_definitions: dict[str, dict[str, str]] = Field(default_factory=dict)
    filter_definitions: dict[str, dict[str, str]] = Field(default_factory=dict)


def vcf_file_kind(path: Path) -> str:
    """Return vcf or vcf.gz based on the complete filename."""
    name = path.name.lower()
    if name.endswith('.vcf.gz'):
        return 'vcf.gz'
    if name.endswith('.vcf'):
        return 'vcf'
    raise VcfPreviewError('VCF preview supports .vcf and .vcf.gz files only.')


def _open_vcf(path: Path) -> TextIO:
    return (
        gzip.open(path, mode='rt', encoding='utf-8', newline='')
        if vcf_file_kind(path) == 'vcf.gz'
        else path.open(encoding='utf-8', newline='')
    )


def _parse_structured_meta(value: str) -> dict[str, str]:
    """Parse comma-separated VCF header attributes while preserving quoted commas."""
    if not value.startswith('<') or not value.endswith('>'):
        return {}
    parts: list[str] = []
    current: list[str] = []
    quoted = False
    escaped = False
    for character in value[1:-1]:
        if escaped:
            current.append(character)
            escaped = False
        elif character == '\\':
            current.append(character)
            escaped = True
        elif character == '"':
            current.append(character)
            quoted = not quoted
        elif character == ',' and not quoted:
            parts.append(''.join(current))
            current = []
        else:
            current.append(character)
    parts.append(''.join(current))
    result: dict[str, str] = {}
    for part in parts:
        if '=' not in part:
            continue
        key, item = part.split('=', 1)
        item = item.strip()
        if len(item) >= QUOTED_VALUE_MIN_LENGTH and item.startswith('"') and item.endswith('"'):
            item = item[1:-1].replace('\\"', '"').replace('\\\\', '\\')
        result[key.strip()] = item
    return result


def _parse_info(text: str) -> dict[str, str | bool]:
    if text in {'', '.'}:
        return {}
    result: dict[str, str | bool] = {}
    for item in text.split(';'):
        if not item:
            continue
        if '=' in item:
            key, value = item.split('=', 1)
            result[key] = value
        else:
            result[item] = True
    return result


def _parse_record(fields: list[str], line_number: int, index: int, expected_columns: int) -> VcfRecord:
    if len(fields) != expected_columns:
        raise VcfPreviewError(
            f'VCF record at line {line_number} has {len(fields)} columns; expected {expected_columns}.'
        )
    chrom, pos_text, identifier, ref, alt_text, qual_text, filter_text, info_text = fields[:REQUIRED_COLUMN_COUNT]
    try:
        pos = int(pos_text)
    except ValueError as error:
        raise VcfPreviewError(f'VCF POS is not an integer at line {line_number}.') from error
    if not chrom or pos < 1 or not ref or not alt_text:
        raise VcfPreviewError(f'VCF record at line {line_number} has an invalid CHROM, POS, REF, or ALT field.')
    if qual_text == '.':
        qual = None
    else:
        try:
            qual = float(qual_text)
        except ValueError as error:
            raise VcfPreviewError(f'VCF QUAL is not numeric at line {line_number}.') from error
        if not math.isfinite(qual):
            raise VcfPreviewError(f'VCF QUAL is not finite at line {line_number}.')
    format_keys = (
        tuple(fields[FORMAT_COLUMN_INDEX].split(':'))
        if len(fields) > FORMAT_COLUMN_INDEX and fields[FORMAT_COLUMN_INDEX] not in {'', '.'}
        else ()
    )
    return VcfRecord(
        index=index,
        line_number=line_number,
        chrom=chrom,
        pos=pos,
        identifier=identifier,
        ref=ref,
        alts=tuple(alt_text.split(',')),
        qual=qual,
        filter_text=filter_text,
        info_text=info_text,
        format_keys=format_keys,
        sample_values=tuple(fields[SAMPLE_COLUMN_START:]),
    )


def read_vcf(path: Path) -> VcfDocument:
    """Load a UTF-8 VCF or gzip-compressed VCF into memory."""
    version = ''
    samples: list[str] = []
    records: list[VcfRecord] = []
    contig_order: list[str] = []
    contig_lengths: dict[str, int] = {}
    info_definitions: dict[str, dict[str, str]] = {}
    format_definitions: dict[str, dict[str, str]] = {}
    filter_definitions: dict[str, dict[str, str]] = {}
    header_columns: list[str] | None = None
    try:
        with _open_vcf(path) as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                line = raw_line.rstrip('\r\n')
                if line_number == 1:
                    line = line.lstrip('\ufeff')
                    if not line.startswith('##fileformat=VCFv'):
                        raise VcfPreviewError('The first VCF line must declare ##fileformat=VCFv…')
                    version = line.split('=', 1)[1]
                    continue
                if not line:
                    continue
                if line.startswith('##'):
                    key, separator, value = line[2:].partition('=')
                    if not separator:
                        continue
                    attributes = _parse_structured_meta(value)
                    identifier = attributes.get('ID')
                    if key == 'contig' and identifier:
                        if identifier not in contig_order:
                            contig_order.append(identifier)
                        length_text = attributes.get('length') or attributes.get('Length')
                        if length_text:
                            with suppress(ValueError):
                                contig_lengths[identifier] = int(length_text)
                    elif key == 'INFO' and identifier:
                        info_definitions[identifier] = attributes
                    elif key == 'FORMAT' and identifier:
                        format_definitions[identifier] = attributes
                    elif key == 'FILTER' and identifier:
                        filter_definitions[identifier] = attributes
                    continue
                if line.startswith('#CHROM'):
                    header_columns = line.split('\t')
                    required = ['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO']
                    if header_columns[:REQUIRED_COLUMN_COUNT] != required:
                        raise VcfPreviewError('VCF column header does not contain the eight required fields in order.')
                    if len(header_columns) > FORMAT_COLUMN_INDEX and header_columns[FORMAT_COLUMN_INDEX] != 'FORMAT':
                        raise VcfPreviewError('The ninth VCF column must be FORMAT when sample columns are present.')
                    samples = header_columns[SAMPLE_COLUMN_START:]
                    continue
                if line.startswith('#'):
                    continue
                if header_columns is None:
                    raise VcfPreviewError('VCF data appears before the #CHROM column header.')
                record = _parse_record(line.split('\t'), line_number, len(records), len(header_columns))
                records.append(record)
                if record.chrom not in contig_order:
                    contig_order.append(record.chrom)
    except VcfPreviewError:
        raise
    except (OSError, UnicodeError) as error:
        raise VcfPreviewError('Could not read this UTF-8 VCF file.') from error
    if not version:
        raise VcfPreviewError('The file does not contain a VCF fileformat declaration.')
    if header_columns is None:
        raise VcfPreviewError('The file does not contain a #CHROM VCF column header.')
    return VcfDocument(
        version=version,
        samples=samples,
        records=records,
        contig_order=contig_order,
        contig_lengths=contig_lengths,
        info_definitions=info_definitions,
        format_definitions=format_definitions,
        filter_definitions=filter_definitions,
    )


def _variant_type(record: VcfRecord) -> str:
    info = _parse_info(record.info_text)
    if 'SVTYPE' in info:
        return 'sv'
    types: set[str] = set()
    for alt in record.alts:
        if alt in {'.', '*'}:
            types.add('other')
        elif alt.startswith('<') or '[' in alt or ']' in alt:
            types.add('sv')
        elif len(record.ref) == len(alt) == 1:
            types.add('snp')
        elif len(record.ref) == len(alt):
            types.add('mnv')
        else:
            types.add('indel')
    return next(iter(types)) if len(types) == 1 else 'mixed'


def _filter_status(record: VcfRecord) -> str:
    if record.filter_text == 'PASS':
        return 'pass'
    if record.filter_text in {'', '.'}:
        return 'unfiltered'
    return 'filtered'


def _transition(record: VcfRecord) -> bool | None:
    if len(record.alts) != 1 or _variant_type(record) != 'snp':
        return None
    pair = f'{record.ref.upper()}{record.alts[0].upper()}'
    if any(base not in {'A', 'C', 'G', 'T'} for base in pair):
        return None
    return pair in {'AG', 'GA', 'CT', 'TC'}


def vcf_metadata(path: Path) -> dict:
    """Return whole-file VCF summaries and per-contig density bins."""
    document = read_vcf(path)
    type_counts: dict[str, int] = dict.fromkeys(('snp', 'mnv', 'indel', 'sv', 'mixed', 'other'), 0)
    filter_counts: dict[str, int] = dict.fromkeys(('pass', 'filtered', 'unfiltered'), 0)
    contig_records: dict[str, list[VcfRecord]] = {name: [] for name in document.contig_order}
    transitions = 0
    transversions = 0
    for record in document.records:
        type_counts[_variant_type(record)] += 1
        filter_counts[_filter_status(record)] += 1
        contig_records.setdefault(record.chrom, []).append(record)
        transition = _transition(record)
        if transition is True:
            transitions += 1
        elif transition is False:
            transversions += 1
    contigs = []
    for name in document.contig_order:
        records = contig_records.get(name, [])
        observed_end = max((record.pos + len(record.ref) - 1 for record in records), default=0)
        length = max(document.contig_lengths.get(name, 0), observed_end)
        bins = [0] * 32
        if length > 0:
            for record in records:
                index = min(len(bins) - 1, (record.pos - 1) * len(bins) // length)
                bins[index] += 1
        contigs.append({'name': name, 'length': length, 'count': len(records), 'bins': bins})
    return {
        'kind': vcf_file_kind(path),
        'version': document.version,
        'file_size': path.stat().st_size,
        'record_count': len(document.records),
        'sample_count': len(document.samples),
        'samples': document.samples,
        'type_counts': type_counts,
        'filter_counts': filter_counts,
        'transitions': transitions,
        'transversions': transversions,
        'ts_tv_ratio': transitions / transversions if transversions else None,
        'contigs': contigs,
        'default_contig': next(
            (record.chrom for record in document.records), document.contig_order[0] if document.contig_order else None
        ),
    }


def _matches_query(
    record: VcfRecord,
    *,
    contig: str | None,
    variant_type: str,
    filter_status: str,
    search: str,
    min_qual: float | None,
) -> bool:
    if contig and record.chrom != contig:
        return False
    if variant_type != 'all' and _variant_type(record) != variant_type:
        return False
    if filter_status != 'all' and _filter_status(record) != filter_status:
        return False
    if min_qual is not None and (record.qual is None or record.qual < min_qual):
        return False
    if search:
        query = search.casefold()
        values = (record.chrom, record.identifier, record.ref, ','.join(record.alts))
        if not any(query in value.casefold() for value in values):
            return False
    return True


def query_vcf(
    path: Path,
    *,
    offset: int,
    limit: int,
    contig: str | None,
    variant_type: str,
    filter_status: str,
    search: str,
    min_qual: float | None,
) -> dict:
    """Filter VCF records and return one page in file order."""
    document = read_vcf(path)
    matching = [
        record
        for record in document.records
        if _matches_query(
            record,
            contig=contig,
            variant_type=variant_type,
            filter_status=filter_status,
            search=search,
            min_qual=min_qual,
        )
    ]
    page = matching[offset : offset + limit]
    return {
        'rows': [
            {
                'record_index': record.index,
                'chrom': record.chrom,
                'pos': record.pos,
                'id': record.identifier,
                'ref': record.ref,
                'alt': ','.join(record.alts),
                'qual': record.qual,
                'filter': record.filter_text,
                'type': _variant_type(record),
            }
            for record in page
        ],
        'offset': offset,
        'limit': limit,
        'total': len(matching),
        'has_more': offset + len(page) < len(matching),
    }


def _genotype_category(genotype: str | None) -> str:
    if not genotype:
        return 'missing'
    alleles = re.split(r'[/|]', genotype)
    if not alleles or any(allele in {'', '.'} for allele in alleles):
        return 'missing'
    if all(allele == '0' for allele in alleles):
        return 'hom_ref'
    if len(set(alleles)) == 1:
        return 'hom_alt'
    return 'het'


def vcf_record_detail(path: Path, record_index: int) -> dict:
    """Return INFO definitions and sample FORMAT values for one VCF record."""
    document = read_vcf(path)
    if record_index >= len(document.records):
        raise VcfPreviewError('The selected VCF record is no longer present.')
    record = document.records[record_index]
    info = _parse_info(record.info_text)
    format_definitions = [
        {
            'key': key,
            'description': document.format_definitions.get(key, {}).get('Description'),
        }
        for key in record.format_keys
    ]
    genotype_counts: dict[str, int] = dict.fromkeys(('hom_ref', 'het', 'hom_alt', 'missing'), 0)
    samples = []
    for sample, raw_values in zip(document.samples, record.sample_values, strict=True):
        values = raw_values.split(':')
        mapped = {key: values[index] if index < len(values) else '.' for index, key in enumerate(record.format_keys)}
        category = _genotype_category(mapped.get('GT'))
        genotype_counts[category] += 1
        samples.append({'sample': sample, 'values': mapped, 'category': category})
    return {
        'record_index': record.index,
        'chrom': record.chrom,
        'pos': record.pos,
        'id': record.identifier,
        'ref': record.ref,
        'alts': list(record.alts),
        'qual': record.qual,
        'filters': [
            {
                'id': value,
                'description': document.filter_definitions.get(value, {}).get('Description'),
            }
            for value in record.filter_text.split(';')
            if value
        ],
        'type': _variant_type(record),
        'info': [
            {
                'key': key,
                'value': value,
                'description': document.info_definitions.get(key, {}).get('Description'),
            }
            for key, value in info.items()
        ],
        'format': format_definitions,
        'samples': samples,
        'genotype_counts': genotype_counts,
    }
