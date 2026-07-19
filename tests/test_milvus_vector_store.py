"""Integration tests for the native Milvus vector store."""

import asyncio
import os
from uuid import uuid4

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from pymilvus import AsyncMilvusClient
from pymilvus.exceptions import MilvusException

from llm.vectorstores import MilvusVectorStore


class TestEmbeddings(Embeddings):
    """Small deterministic embedding model for Milvus integration tests."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents into a stable two-dimensional space."""
        return [[1.0, 0.0] if 'alpha' in text else [0.0, 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a query into the same stable space."""
        return [1.0, 0.0] if 'alpha' in text else [0.0, 1.0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents without a thread-pool fallback."""
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        """Embed a query without a thread-pool fallback."""
        return self.embed_query(text)


def test_native_vector_store_against_isolated_milvus_database() -> None:
    """Exercise dense, BM25 hybrid, status, and LangChain retriever integration."""
    asyncio.run(_run_milvus_integration_test())


async def _run_milvus_integration_test() -> None:
    uri = os.getenv('MILVUS_TEST_URI', 'http://localhost:19530')
    database_name = f'bioflow_test_{uuid4().hex}'
    collection_name = 'VectorStoreContract'

    try:
        async with AsyncMilvusClient(uri=uri, timeout=5) as admin:
            await admin.create_database(database_name)
    except MilvusException as exc:
        pytest.skip(f'Milvus is unavailable at {uri}: {exc}')

    try:
        store = MilvusVectorStore(
            embedding_function=TestEmbeddings(),
            collection_name=collection_name,
            uri=uri,
            db_name=database_name,
            hybrid=True,
            timeout=5,
        )
        assert isinstance(store, VectorStore)
        retriever = store.as_retriever(search_kwargs={'k': 1, 'weights': (0.85, 0.15)})
        assert retriever.vectorstore is store

        ids = await store.aadd_texts(
            ['alpha sequence alignment', 'beta differential expression'],
            [{'kind': 'alignment'}, {'kind': 'expression'}],
        )
        documents = await store.asimilarity_search('alpha alignment', k=2, weights=(0.85, 0.15))
        retrieved_documents = await retriever.ainvoke('alpha alignment')
        diverse_documents = await store.amax_marginal_relevance_search('alpha alignment', k=1)
        status = await store.status()

        assert len(ids) == len(('alpha', 'beta'))
        assert documents
        assert documents[0].page_content == 'alpha sequence alignment'
        assert documents[0].metadata['kind'] == 'alignment'
        assert retrieved_documents[0].page_content == 'alpha sequence alignment'
        assert diverse_documents[0].page_content == 'alpha sequence alignment'
        assert status.exists
        assert status.row_count == len(ids)
    finally:
        async with AsyncMilvusClient(uri=uri, db_name=database_name, timeout=5) as database:
            if await database.has_collection(collection_name):
                await database.drop_collection(collection_name)
        async with AsyncMilvusClient(uri=uri, timeout=5) as admin:
            await admin.drop_database(database_name)
