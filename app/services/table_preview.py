"""Direct Polars-backed CSV and TSV previews."""

# Stable UI-facing parser messages are intentionally raised at their call sites.
# ruff: noqa: EM101, EM102, TRY003

import math
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, NoReturn

import polars as pl


class TablePreviewError(ValueError):
    """A CSV/TSV preview request that cannot be evaluated."""


def table_file_kind(path: Path) -> str:
    """Return the delimiter-defined table kind."""
    suffix = path.suffix.lower()
    if suffix in {'.csv', '.tsv'}:
        return suffix[1:]
    raise TablePreviewError('Table preview supports CSV and TSV files only.')


def _read_table(path: Path) -> tuple[pl.DataFrame, str]:
    kind = table_file_kind(path)
    try:
        frame = pl.read_csv(
            path,
            separator=',' if kind == 'csv' else '\t',
            try_parse_dates=True,
            infer_schema_length=1_000,
            encoding='utf8',
        )
    except (OSError, UnicodeError, pl.exceptions.PolarsError) as error:
        raise TablePreviewError(f'File is not valid UTF-8 {kind.upper()} data.') from error
    return frame, kind


def _data_kind(dtype: pl.DataType) -> str:
    name = str(dtype).lower()
    if name.startswith(('int', 'uint')):
        return 'integer'
    if name.startswith(('float', 'decimal')):
        return 'number'
    if name == 'boolean':
        return 'boolean'
    if name.startswith(('date', 'datetime', 'time')):
        return 'temporal'
    if name == 'null':
        return 'null'
    return 'text'


def table_metadata(path: Path) -> dict[str, Any]:
    """Read a table and describe its rows, columns, and inferred types."""
    frame, kind = _read_table(path)
    return {
        'kind': kind,
        'supported': True,
        'file_size': path.stat().st_size,
        'row_count': frame.height,
        'delimiter': ',' if kind == 'csv' else '\t',
        'columns': [
            {'name': name, 'dtype': str(dtype), 'kind': _data_kind(dtype)} for name, dtype in frame.schema.items()
        ],
    }


def _raise_invalid_filter(dtype: pl.DataType, error: Exception | None = None) -> NoReturn:
    raise TablePreviewError(f'Filter value is not valid for {dtype}.') from error


def _filter_value(dtype: pl.DataType, value: str | None) -> Any:
    if value is None:
        raise TablePreviewError('This filter requires a value.')
    kind = _data_kind(dtype)
    name = str(dtype).lower()
    try:
        if kind == 'integer':
            return int(value)
        if name.startswith('decimal'):
            return Decimal(value)
        if kind == 'number':
            result = float(value)
            if not math.isfinite(result):
                _raise_invalid_filter(dtype)
            return result
        if kind == 'boolean':
            normalized = value.strip().lower()
            if normalized in {'true', '1', 'yes'}:
                return True
            if normalized in {'false', '0', 'no'}:
                return False
            _raise_invalid_filter(dtype)
        if name == 'date':
            return date.fromisoformat(value)
        if name.startswith('datetime'):
            return datetime.fromisoformat(value)
        if name == 'time':
            return time.fromisoformat(value)
    except (InvalidOperation, ValueError) as error:
        _raise_invalid_filter(dtype, error)
    return value


def _apply_filters(frame: pl.DataFrame, filters: list[dict[str, Any]]) -> pl.DataFrame:
    expressions = []
    for item in filters:
        name = item['column']
        if name not in frame.schema:
            raise TablePreviewError('A filter references a column that is not present.')
        operator = item['operator']
        column = pl.col(name)
        if operator == 'is_null':
            expressions.append(column.is_null())
            continue
        if operator == 'is_not_null':
            expressions.append(column.is_not_null())
            continue
        value = _filter_value(frame.schema[name], item.get('value'))
        if operator == 'contains':
            expressions.append(column.cast(pl.String).str.contains(str(value), literal=True).fill_null(value=False))
        elif operator == 'equals':
            expressions.append((column == value).fill_null(value=False))
        elif operator == 'not_equals':
            expressions.append((column != value).fill_null(value=False))
        elif operator in {'gt', 'gte', 'lt', 'lte'}:
            if _data_kind(frame.schema[name]) not in {'integer', 'number', 'temporal'}:
                raise TablePreviewError(f'{operator} is not valid for {frame.schema[name]}.')
            comparisons = {
                'gt': column > value,
                'gte': column >= value,
                'lt': column < value,
                'lte': column <= value,
            }
            expressions.append(comparisons[operator].fill_null(value=False))
        else:
            raise TablePreviewError('Unknown table filter operator.')
    return frame.filter(*expressions) if expressions else frame


def _cell(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    return value


def query_table(
    path: Path,
    *,
    offset: int,
    limit: int,
    columns: list[str],
    sort: dict[str, Any] | None,
    filters: list[dict[str, Any]],
) -> dict[str, Any]:
    """Load a table, apply simple operations, and return one page."""
    frame, _ = _read_table(path)
    if len(columns) != len(set(columns)):
        raise TablePreviewError('Selected columns must be distinct.')
    if any(column not in frame.schema for column in columns):
        raise TablePreviewError('A selected column is not present in this table.')
    frame = _apply_filters(frame, filters)
    if sort:
        if sort['column'] not in frame.schema:
            raise TablePreviewError('The sort column is not present in this table.')
        frame = frame.sort(sort['column'], descending=sort['direction'] == 'desc', nulls_last=True)
    total = frame.height
    result = frame.select(columns).slice(offset, limit)
    return {
        'columns': columns,
        'rows': [[_cell(value) for value in row] for row in result.rows()],
        'offset': offset,
        'limit': limit,
        'total': total,
        'has_more': offset + result.height < total,
    }


def profile_column(path: Path, *, column: str, filters: list[dict[str, Any]]) -> dict[str, Any]:
    """Load a table and summarize one column after filtering."""
    frame, _ = _read_table(path)
    if column not in frame.schema:
        raise TablePreviewError('The profile column is not present in this table.')
    frame = _apply_filters(frame, filters)
    data_kind = _data_kind(frame.schema[column])
    expressions = [
        pl.len().alias('count'),
        pl.col(column).null_count().alias('null_count'),
        pl.col(column).n_unique().alias('unique_count'),
    ]
    if data_kind in {'integer', 'number'}:
        expressions.extend(
            [
                pl.col(column).min().alias('minimum'),
                pl.col(column).max().alias('maximum'),
                pl.col(column).mean().alias('mean'),
            ]
        )
    elif data_kind == 'temporal':
        expressions.extend(
            [
                pl.col(column).min().alias('minimum'),
                pl.col(column).max().alias('maximum'),
            ]
        )
    row = frame.select(expressions).row(0, named=True)
    return {
        'column': column,
        'dtype': str(frame.schema[column]),
        'kind': data_kind,
        **{name: _cell(value) for name, value in row.items()},
    }
