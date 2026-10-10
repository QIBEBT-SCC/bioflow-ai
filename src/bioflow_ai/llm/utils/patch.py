"""Context-hunk parsing and text updates shared by CODE editing tools."""

from pydantic import BaseModel, Field

MAX_PATCH_CHARS = 1_000_000
UNICODE_ASCII = str.maketrans(
    '\u2010\u2011\u2012\u2013\u2014\u2212\u2018\u2019\u201a\u201c\u201d\u201e\u00a0\u2002\u2003\u2009',
    '------\'\'\'"""    ',
)


class PatchError(ValueError):
    """An invalid patch or a failed patch operation."""


class PatchChunk(BaseModel):
    context: str | None = None
    old_lines: list[str] = Field(default_factory=list)
    new_lines: list[str] = Field(default_factory=list)
    # Indices of corresponding old/new context lines, preserved after fuzzy matching.
    context_indices: list[tuple[int, int]] = Field(default_factory=list)
    end_of_file: bool = False


def parse_patch_chunks(lines: list[str], path: str) -> list[PatchChunk]:
    """Parse context hunks for one source without handling file operations."""
    chunks: list[PatchChunk] = []
    chunk: PatchChunk | None = None
    for index, line in enumerate(lines):
        if line == '@@' or line.startswith('@@ '):
            chunk = PatchChunk(context=line[3:] if line.startswith('@@ ') else None)
            chunks.append(chunk)
        elif line == '*** End of File':
            if chunk is None or index != len(lines) - 1:
                msg = f'Update File {path}: End of File must end the final chunk.'
                raise PatchError(msg)
            chunk.end_of_file = True
        elif line and line[0] in ' +-':
            if chunk is None:
                chunk = PatchChunk()
                chunks.append(chunk)
            prefix, content = line[0], line[1:]
            if prefix == ' ':
                chunk.context_indices.append((len(chunk.old_lines), len(chunk.new_lines)))
            if prefix in ' -':
                chunk.old_lines.append(content)
            if prefix in ' +':
                chunk.new_lines.append(content)
        else:
            msg = f'Update File {path}: invalid change line {line!r}.'
            raise PatchError(msg)
    if not chunks or any(not chunk.old_lines and not chunk.new_lines for chunk in chunks):
        msg = f'Update File {path}: an update must contain nonempty change chunks.'
        raise PatchError(msg)
    return chunks


def _seek(lines: list[str], expected: list[str], start: int, *, end_of_file: bool = False) -> int | None:
    if not expected:
        return len(lines)
    last = len(lines) - len(expected)
    positions = [last] if end_of_file else range(start, last + 1)
    for normalize in (str, str.rstrip, str.strip, lambda text: text.strip().translate(UNICODE_ASCII)):
        for position in positions:
            if position >= start and all(
                normalize(lines[position + offset]) == normalize(value) for offset, value in enumerate(expected)
            ):
                return position
    return None


def derive_new_contents(source: str, chunks: list[PatchChunk], path: str) -> str:
    """Match hunks forwards, preserve actual context, and normalize output to LF."""
    lines = source.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    if lines[-1] == '':
        lines.pop()
    replacements: list[tuple[int, int, list[str]]] = []
    cursor = 0
    for chunk in chunks:
        if chunk.context is not None:
            context_index = _seek(lines, [chunk.context], cursor)
            if context_index is None:
                msg = f'Failed to find context {chunk.context!r} in {path}.'
                raise PatchError(msg)
            cursor = context_index + 1
        expected = chunk.old_lines
        replacement = list(chunk.new_lines)
        position = _seek(lines, expected, cursor, end_of_file=chunk.end_of_file)
        if position is None and expected and expected[-1] == '':
            expected = expected[:-1]
            if replacement and replacement[-1] == '':
                replacement = replacement[:-1]
            position = _seek(lines, expected, cursor, end_of_file=chunk.end_of_file)
        if position is None:
            msg = f'Failed to find expected lines in {path}: {chunk.old_lines!r}'
            raise PatchError(msg)
        for old_index, new_index in chunk.context_indices:
            if old_index < len(expected) and new_index < len(replacement):
                replacement[new_index] = lines[position + old_index]
        replacements.append((position, len(expected), replacement))
        cursor = position + len(expected)
    for position, length, replacement in reversed(replacements):
        lines[position : position + length] = replacement
    return '\n'.join(lines) + ('\n' if lines else '')
