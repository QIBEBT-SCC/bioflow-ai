"""Native asynchronous Milvus vector store integration.

This module deliberately keeps the small subset of vector-store behaviour used by
BioFlowAI instead of depending on ``langchain-milvus``. LangChain is still used for
the portable ``Document`` and ``Embeddings`` interfaces.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

import numpy as np
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores.utils import maximal_marginal_relevance
from pymilvus import (
    AnnSearchRequest,
    AsyncMilvusClient,
    DataType,
    Function,
    FunctionType,
    WeightedRanker,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable
    from types import TracebackType

    from langchain_core.embeddings import Embeddings


@dataclass(frozen=True)
class MilvusCollectionStatus:
    """Operational information for a Milvus collection."""

    name: str
    exists: bool
    loaded: bool = False
    row_count: int | None = None
    description: dict[str, Any] | None = None


class MilvusVectorStore(VectorStore):
    """A focused, natively asynchronous Milvus vector store.

    Collections are created lazily on the first insert because the vector dimension
    is only known after embedding the first batch. Existing collections created by
    ``langchain-milvus`` remain compatible: field names and schemas intentionally
    use the same ``pk``, ``text``, ``vector``/``dense`` and ``sparse`` convention.
    """

    _PRIMARY_FIELD = 'pk'
    _TEXT_FIELD = 'text'
    _DENSE_FIELD = 'dense'
    _SINGLE_VECTOR_FIELD = 'vector'
    _SPARSE_FIELD = 'sparse'

    def __init__(
        self,
        *,
        embedding_function: Embeddings,
        collection_name: str,
        uri: str,
        db_name: str = '',
        hybrid: bool = False,
        drop_old: bool = False,
        auto_id: bool = True,
        consistency_level: str = 'Strong',
        metric_type: str = 'L2',
        index_type: str = 'HNSW',
        index_params: dict[str, Any] | None = None,
        search_params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> None:
        self.embedding_function = embedding_function
        self.collection_name = collection_name
        self.uri = uri
        self.db_name = db_name
        self.hybrid = hybrid
        self.drop_old = drop_old
        self.auto_id = auto_id
        self.consistency_level = consistency_level
        self.metric_type = metric_type
        self.index_type = index_type
        self.index_params = index_params or {'M': 64, 'efConstruction': 100}
        self.search_params = search_params or {'ef': 64}
        self.timeout = timeout
        self._client: AsyncMilvusClient | None = None
        self._drop_pending = drop_old

    @property
    def dense_field(self) -> str:
        """Return the dense-vector field used by this collection."""
        return self._DENSE_FIELD if self.hybrid else self._SINGLE_VECTOR_FIELD

    async def __aenter__(self) -> Self:
        """Open a native asynchronous Milvus client."""
        self._client = await AsyncMilvusClient(uri=self.uri, db_name=self.db_name, timeout=self.timeout).__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the native asynchronous Milvus client."""
        await self.close()

    async def close(self) -> None:
        """Close the underlying asynchronous Milvus connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    @asynccontextmanager
    async def _client_scope(self) -> AsyncGenerator[AsyncMilvusClient]:
        """Reuse an explicit context client or create one for a single operation."""
        if self._client is not None:
            yield self._client
            return
        async with AsyncMilvusClient(uri=self.uri, db_name=self.db_name, timeout=self.timeout) as client:
            yield client

    async def _prepare_collection(self, client: AsyncMilvusClient, dimension: int | None = None) -> bool:
        exists = await client.has_collection(self.collection_name)
        if self._drop_pending and exists:
            await client.drop_collection(self.collection_name)
            exists = False
        self._drop_pending = False

        if exists:
            load_state = await client.get_load_state(self.collection_name)
            if not str(load_state.get('state', '')).lower().endswith('loaded'):
                await client.load_collection(self.collection_name)
            return True
        if dimension is None:
            return False

        schema = client.create_schema(auto_id=self.auto_id, enable_dynamic_field=True)
        text_kwargs = {'enable_analyzer': True} if self.hybrid else {}
        schema.add_field(
            field_name=self._TEXT_FIELD,
            datatype=DataType.VARCHAR,
            max_length=65_535,
            **text_kwargs,
        )
        schema.add_field(
            field_name=self._PRIMARY_FIELD,
            datatype=DataType.INT64 if self.auto_id else DataType.VARCHAR,
            is_primary=True,
            auto_id=self.auto_id,
            **({} if self.auto_id else {'max_length': 65_535}),
        )
        schema.add_field(field_name=self.dense_field, datatype=DataType.FLOAT_VECTOR, dim=dimension)

        if self.hybrid:
            schema.add_field(
                field_name=self._SPARSE_FIELD,
                datatype=DataType.SPARSE_FLOAT_VECTOR,
                is_function_output=True,
            )
            schema.add_function(
                Function(
                    name='text_bm25',
                    function_type=FunctionType.BM25,
                    input_field_names=self._TEXT_FIELD,
                    output_field_names=self._SPARSE_FIELD,
                )
            )

        indexes = client.prepare_index_params()
        indexes.add_index(
            field_name=self.dense_field,
            index_type=self.index_type,
            metric_type=self.metric_type,
            params=self.index_params,
        )
        if self.hybrid:
            indexes.add_index(
                field_name=self._SPARSE_FIELD,
                index_type='AUTOINDEX',
                metric_type='BM25',
                params={},
            )

        await client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=indexes,
            consistency_level=self.consistency_level,
        )
        return True

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> Self:
        """Create a store and insert texts through the synchronous API."""
        store = cls(embedding_function=embedding, **kwargs)
        store.add_texts(texts, metadatas, ids=ids)
        return store

    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        """Insert documents from synchronous code."""
        import asyncio

        return asyncio.run(self.aadd_documents(documents, **kwargs))

    async def aadd_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        """Embed and insert documents without blocking the event loop."""
        ids: list[str] | None = kwargs.pop('ids', None)
        if kwargs:
            msg = f'Unsupported insert options: {", ".join(kwargs)}'
            raise TypeError(msg)
        if not documents:
            return []
        if not self.auto_id and (ids is None or len(ids) != len(documents)):
            msg = 'ids must match documents when auto_id is disabled'
            raise ValueError(msg)

        texts = [document.page_content for document in documents]
        embeddings = await self.embedding_function.aembed_documents(texts)
        if len(embeddings) != len(documents) or not embeddings[0]:
            msg = 'Embedding provider returned an invalid number of vectors'
            raise ValueError(msg)

        rows: list[dict[str, Any]] = []
        for index, (document, embedding) in enumerate(zip(documents, embeddings, strict=True)):
            row = {self._TEXT_FIELD: document.page_content, self.dense_field: embedding, **document.metadata}
            if ids is not None and not self.auto_id:
                row[self._PRIMARY_FIELD] = ids[index]
            rows.append(row)

        async with self._client_scope() as client:
            await self._prepare_collection(client, len(embeddings[0]))
            result = await client.insert(collection_name=self.collection_name, data=rows)
        primary_keys = result.get('ids', [])
        return [str(primary_key) for primary_key in primary_keys]

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        """Search from synchronous code."""
        import asyncio

        return asyncio.run(self.asimilarity_search(query, k=k, **kwargs))

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        """Search by dense similarity or dense/BM25 weighted hybrid similarity."""
        results = await self.asimilarity_search_with_score(query, k=k, **kwargs)
        return [document for document, _score in results]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Search with raw Milvus distance scores from synchronous code."""
        import asyncio

        return asyncio.run(self.asimilarity_search_with_score(query, k=k, **kwargs))

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """Select LangChain's score normalization for the configured metric."""
        if self.hybrid:
            return float
        match self.metric_type.upper():
            case 'COSINE':
                return self._cosine_relevance_score_fn
            case 'IP':
                return self._max_inner_product_relevance_score_fn
            case _:
                return self._euclidean_relevance_score_fn

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Search with raw Milvus distance scores."""
        filter_expression: str = kwargs.pop('filter_expression', kwargs.pop('expr', ''))
        weights: tuple[float, float] = kwargs.pop('weights', (0.85, 0.15))
        if kwargs:
            msg = f'Unsupported search options: {", ".join(kwargs)}'
            raise TypeError(msg)
        query_embedding = await self.embedding_function.aembed_query(query)
        output_fields = ['*']
        async with self._client_scope() as client:
            if not await self._prepare_collection(client):
                return []
            if self.hybrid:
                requests = [
                    AnnSearchRequest(
                        data=[query_embedding],
                        anns_field=self.dense_field,
                        param={'metric_type': self.metric_type, 'params': self.search_params},
                        limit=k,
                        expr=filter_expression or None,
                    ),
                    AnnSearchRequest(
                        data=[query],
                        anns_field=self._SPARSE_FIELD,
                        param={'metric_type': 'BM25', 'params': {}},
                        limit=k,
                        expr=filter_expression or None,
                    ),
                ]
                results = await client.hybrid_search(
                    collection_name=self.collection_name,
                    reqs=requests,
                    ranker=WeightedRanker(*weights),
                    limit=k,
                    output_fields=output_fields,
                )
            else:
                results = await client.search(
                    collection_name=self.collection_name,
                    data=[query_embedding],
                    anns_field=self.dense_field,
                    search_params={'metric_type': self.metric_type, 'params': self.search_params},
                    filter=filter_expression,
                    limit=k,
                    output_fields=output_fields,
                )
        return self._documents_with_scores_from_results(results)

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        """Run MMR search from synchronous code."""
        import asyncio

        return asyncio.run(
            self.amax_marginal_relevance_search(
                query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                **kwargs,
            )
        )

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        """Return dense search results reordered for relevance and diversity."""
        filter_expression: str = kwargs.pop('filter_expression', kwargs.pop('expr', ''))
        if kwargs:
            msg = f'Unsupported MMR options: {", ".join(kwargs)}'
            raise TypeError(msg)
        query_embedding = await self.embedding_function.aembed_query(query)
        async with self._client_scope() as client:
            if not await self._prepare_collection(client):
                return []
            results = await client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                anns_field=self.dense_field,
                search_params={'metric_type': self.metric_type, 'params': self.search_params},
                filter=filter_expression,
                limit=fetch_k,
                output_fields=['*', self.dense_field],
            )
        hits = results[0] if results else []
        documents = [self._document_from_hit(hit) for hit in hits]
        vectors = [hit['entity'][self.dense_field] for hit in hits]
        ordering = maximal_marginal_relevance(
            np.asarray(query_embedding),
            vectors,
            lambda_mult=lambda_mult,
            k=k,
        )
        return [documents[index] for index in ordering]

    async def status(self) -> MilvusCollectionStatus:
        """Return collection state suitable for a future management API."""
        async with self._client_scope() as client:
            if not await client.has_collection(self.collection_name):
                return MilvusCollectionStatus(name=self.collection_name, exists=False)

            description = await client.describe_collection(self.collection_name)
            load_state = await client.get_load_state(self.collection_name)
            stats = await client.get_collection_stats(self.collection_name)
            counts = await client.query(
                collection_name=self.collection_name,
                filter='',
                output_fields=['count(*)'],
                consistency_level=self.consistency_level,
            )
        row_count = counts[0].get('count(*)') if counts else stats.get('row_count')
        return MilvusCollectionStatus(
            name=self.collection_name,
            exists=True,
            loaded=str(load_state.get('state', '')).lower().endswith('loaded'),
            row_count=int(row_count) if row_count is not None else None,
            description=description,
        )

    def _documents_with_scores_from_results(
        self,
        results: list[list[dict[str, Any]]],
    ) -> list[tuple[Document, float]]:
        if not results:
            return []
        return [(self._document_from_hit(hit), float(hit['distance'])) for hit in results[0]]

    def _document_from_hit(self, hit: dict[str, Any]) -> Document:
        entity = dict(hit.get('entity', {}))
        page_content = entity.pop(self._TEXT_FIELD, '')
        for field in (self._PRIMARY_FIELD, self.dense_field, self._SPARSE_FIELD):
            entity.pop(field, None)
        return Document(page_content=page_content, metadata=entity)
