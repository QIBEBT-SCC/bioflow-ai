"""Lossless ordered Markdown blocks and paragraph-level relevance rendering."""

import hashlib
from html import escape

from markdown_it import MarkdownIt
from pydantic import BaseModel, Field

SPLIT_VERSION = 1
FRAGMENT_CHARS = 2000


class MarkdownBlock(BaseModel):
    """A source span with enclosing heading IDs; duplicate text keeps distinct IDs."""

    index: int
    start: int
    end: int
    heading: bool = False
    parents: list[int] = Field(default_factory=list)


class PDFManifest(BaseModel):
    """The ordered source map and the exact set of vectors needed for a document."""

    version: int = SPLIT_VERSION
    markdown_key: str
    text_hash: str
    embedding_fingerprint: str
    blocks: list[MarkdownBlock]
    vector_ids: list[str]
    vector_blocks: list[int]


def text_hash(text: str) -> str:
    """Hash the exact stored Markdown, including whitespace."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def split_markdown(text: str) -> list[MarkdownBlock]:
    """Keep top-level Markdown structures intact and retain their source spans."""
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    tokens = MarkdownIt('commonmark').enable('table').parse(text)
    blocks: list[MarkdownBlock] = []
    headings: list[tuple[int, int]] = []
    consumed_line = -1
    for token in tokens:
        if token.level != 0 or token.map is None or token.map[0] < consumed_line:
            continue
        start_line, end_line = token.map
        consumed_line = end_line
        is_heading = token.type == 'heading_open'
        if is_heading:
            level = int(token.tag[1:])
            while headings and headings[-1][0] >= level:
                headings.pop()
        index = len(blocks)
        blocks.append(
            MarkdownBlock(
                index=index,
                start=offsets[start_line],
                end=offsets[end_line],
                heading=is_heading,
                parents=[heading_index for _, heading_index in headings],
            )
        )
        if is_heading:
            headings.append((level, index))
    # Assign gaps to the preceding block so rendering retains original whitespace.
    for index, block in enumerate(blocks):
        block.end = blocks[index + 1].start if index + 1 < len(blocks) else len(text)
    if blocks:
        blocks[0].start = 0
    return blocks


def embedding_fragments(text: str, blocks: list[MarkdownBlock]) -> tuple[list[str], list[int]]:
    """Split only retrieval inputs; output remains the entire original block."""
    fragments: list[str] = []
    indexes: list[int] = []
    for block in blocks:
        if block.heading:
            continue
        body = text[block.start : block.end].strip()
        for start in range(0, len(body), FRAGMENT_CHARS):
            fragments.append(body[start : start + FRAGMENT_CHARS])
            indexes.append(block.index)
    # Heading-only documents must still have a usable index.
    if not fragments:
        for block in blocks:
            fragments.append(text[block.start : block.end].strip())
            indexes.append(block.index)
    return fragments, indexes


def render_filtered(text: str, blocks: list[MarkdownBlock], retained: set[int]) -> str:
    """Render hits and their ancestors in source order, merging omission runs."""
    if not retained:
        return 'No paragraphs exceeded the configured similarity threshold.'
    visible = set(retained)
    for block in blocks:
        if block.index in retained:
            visible.update(block.parents)
    result: list[str] = []
    skipped = 0
    for block in blocks:
        if block.index not in visible:
            skipped += int(not block.heading)
            continue
        if skipped:
            result.append(f'…（已省略 {skipped} 段）\n\n')  # noqa: RUF001 - requested omission label
            skipped = 0
        result.append(text[block.start : block.end])
    if skipped:
        result.append(f'…（已省略 {skipped} 段）\n')  # noqa: RUF001 - requested omission label
    return ''.join(result)


def pdf_block(file_id: str, text: str) -> str:
    """Escape source text so documents cannot break out of their XML wrapper."""
    return f'<PDF id="{escape(file_id, quote=True)}">\n{escape(text, quote=False)}\n</PDF>'
