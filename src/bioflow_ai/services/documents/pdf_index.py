"""Prepare existing parser output for PDF attachments and cosine retrieval."""

import asyncio
import base64
import hashlib
import json
import re
from pathlib import Path, PurePosixPath

import anydoc
import numpy as np
from pymilvus import AsyncMilvusClient, DataType
from sqlalchemy import select
from sqlalchemy.orm import Session

from bioflow_ai.core.config import get_settings
from bioflow_ai.infrastructure.database import engine
from bioflow_ai.infrastructure.object_storage import files as storage
from bioflow_ai.llm.models import pdf_embedding
from bioflow_ai.models.resource import File
from bioflow_ai.schemas.agent import MAX_PDF_BYTES, PDFMessagePart
from bioflow_ai.services.documents.paragraphs import (
    SPLIT_VERSION,
    PDFManifest,
    embedding_fragments,
    pdf_block,
    render_filtered,
    split_markdown,
    text_hash,
)
from bioflow_ai.services.documents.pdf import parse_pdf

COLLECTION = 'PDFParagraphs'
QUERY_INSTRUCTION = 'Given a query about a scientific paper, retrieve relevant passages that answer the query'
READ_BATCH = 256


class PDFNotFoundError(ValueError):
    """The referenced file is absent from the file registry."""


class PDFPreparationError(ValueError):
    """A recoverable attachment validation or indexing failure."""


def embedding_fingerprint() -> str:
    """Identify the embedding space without recording credentials."""
    config = get_settings().knowledge_base
    return hashlib.sha256(json.dumps([config.BASE_URL, config.EMBEDDING_MODEL]).encode()).hexdigest()


def _normalize(values: list[list[float]]) -> np.ndarray:
    vectors = np.asarray(values, dtype=np.float64)
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


async def build_index(text: str, document_hash: str, markdown_key: str) -> PDFManifest:
    """Upsert deterministic fragment IDs so partial preparations can be retried."""
    blocks = split_markdown(text)
    fragments, block_ids = embedding_fragments(text, blocks)
    if not fragments:
        message = 'The parsed PDF contains no indexable text.'
        raise ValueError(message)
    digest = text_hash(text)
    vector_ids = [f'{document_hash}:{digest}:{index}' for index in range(len(fragments))]
    vectors = _normalize(await pdf_embedding().aembed_documents(fragments))
    async with AsyncMilvusClient(uri=get_settings().knowledge_base.MILVUS_URL) as client:
        if not await client.has_collection(COLLECTION):
            schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field('pk', DataType.VARCHAR, is_primary=True, max_length=160)
            schema.add_field('vector', DataType.FLOAT_VECTOR, dim=vectors.shape[1])
            indexes = client.prepare_index_params()
            indexes.add_index(field_name='vector', index_type='FLAT', metric_type='COSINE')
            await client.create_collection(
                collection_name=COLLECTION, schema=schema, index_params=indexes, consistency_level='Strong'
            )
        await client.upsert(
            collection_name=COLLECTION,
            data=[{'pk': uid, 'vector': vector.tolist()} for uid, vector in zip(vector_ids, vectors, strict=True)],
        )
    return PDFManifest(
        markdown_key=markdown_key,
        text_hash=digest,
        embedding_fingerprint=embedding_fingerprint(),
        blocks=blocks,
        vector_ids=vector_ids,
        vector_blocks=block_ids,
    )


async def relevant_blocks(manifest: PDFManifest, query: str) -> set[int]:
    """Score every indexed fragment; no ANN/top-k cap can discard eligible blocks."""
    query_input = f'Instruct: {QUERY_INSTRUCTION}\nQuery: {query.strip()}'
    vector = _normalize([await pdf_embedding().aembed_query(query_input)])[0]
    threshold = get_settings().pdf_parser.SIMILARITY_THRESHOLD
    retained: set[int] = set()
    async with AsyncMilvusClient(uri=get_settings().knowledge_base.MILVUS_URL) as client:
        for offset in range(0, len(manifest.vector_ids), READ_BATCH):
            ids = manifest.vector_ids[offset : offset + READ_BATCH]
            rows = await client.get(collection_name=COLLECTION, ids=ids, output_fields=['vector'])
            by_id = {row['pk']: row['vector'] for row in rows}
            scores = _normalize([by_id[uid] for uid in ids]) @ vector
            retained.update(
                manifest.vector_blocks[offset + index] for index, score in enumerate(scores) if score > threshold
            )
    return retained


def _manifest_key(digest: str) -> str:
    return f'documents/{digest}/index.json'


def _prepared_manifest(digest: str) -> PDFManifest | None:
    """Return the stored manifest only when it matches the current split and embedding setup."""
    key = _manifest_key(digest)
    if not storage.file_exists(key):
        return None
    manifest = PDFManifest.model_validate_json(storage.read_text_file(key))
    if manifest.version != SPLIT_VERSION or manifest.embedding_fingerprint != embedding_fingerprint():
        return None
    return manifest


def _find_file(file_id: str) -> File:
    with Session(engine) as session:
        file = session.scalar(select(File).where(File.s3_path == file_id))
    if file is None:
        message = 'PDF file not found.'
        raise PDFNotFoundError(message)
    return file


def _message_part(file: File) -> PDFMessagePart:
    return PDFMessagePart(id=file.id, file_id=file.s3_path, filename=file.file_name)


async def prepare_pdf_attachment(data: bytes, filename: str) -> PDFMessagePart:
    """Call the existing parser, then index its Markdown for agent retrieval."""
    if not data or len(data) > MAX_PDF_BYTES:
        message = 'PDF must be non-empty and no larger than 20 MiB.'
        raise PDFPreparationError(message)
    if anydoc.format_from_bytes(data) != 'pdf':
        message = 'The uploaded file is not a valid PDF.'
        raise PDFPreparationError(message)
    filename = PurePosixPath(filename.replace('\\', '/')).name[:255] or 'document.pdf'
    digest = hashlib.sha256(data).hexdigest()
    try:
        file = await asyncio.to_thread(_find_file, await parse_pdf(base64.b64encode(data).decode('ascii'), filename))
        if await asyncio.to_thread(_prepared_manifest, digest) is None:
            # Reuse the parser's Markdown artifact; keep image captions as plain text for the model.
            parser_key = storage.generate_key_from_hash(
                file.sha256, Path(file.file_name).with_suffix('.md').name, 'documents'
            )
            text = await asyncio.to_thread(storage.read_text_file, parser_key)
            text = re.sub(r'!\[([^\]]*)\]\([^\n]*?\)', r'\1', text)
            markdown_key = f'documents/{digest}/{text_hash(text)}.md'
            await asyncio.to_thread(storage.write_bytes, markdown_key, text.encode(), 'text/markdown; charset=utf-8')
            manifest = await build_index(text, digest, markdown_key)
            await asyncio.to_thread(
                storage.write_bytes, _manifest_key(digest), manifest.model_dump_json().encode(), 'application/json'
            )
        return _message_part(file)
    except storage.ObjectStorageError:
        raise
    except Exception as error:
        raise PDFPreparationError(str(error) or type(error).__name__) from error


def resolve_pdf(session: Session, part: PDFMessagePart) -> PDFMessagePart:
    """Validate a registered PDF and its prepared index before sending a message."""
    file = session.get(File, part.id)
    if file is None:
        message = 'PDF file not found.'
        raise PDFNotFoundError(message)
    if file.s3_path != part.file_id:
        message = 'PDF file ID does not match its file record.'
        raise PDFPreparationError(message)
    if _prepared_manifest(file.sha256) is None:
        message = 'PDF text and index must be prepared before sending. Upload the PDF again.'
        raise PDFPreparationError(message)
    return _message_part(file)


def _read_text(file_id: str) -> tuple[str, PDFManifest]:
    manifest = _prepared_manifest(_find_file(file_id).sha256)
    if manifest is None:
        message = 'The PDF index is unavailable; prepare the attachment again.'
        raise PDFPreparationError(message)
    return storage.read_text_file(manifest.markdown_key), manifest


async def read_pdf(file_id: str, query: str | None = None) -> str:
    """Read a registered PDF as complete text or ordered threshold-filtered text."""
    text, manifest = await asyncio.to_thread(_read_text, file_id)
    if query and query.strip():
        text = render_filtered(text, manifest.blocks, await relevant_blocks(manifest, query))
    return pdf_block(file_id, text)
