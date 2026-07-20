import asyncio
from types import SimpleNamespace
from typing import ClassVar, Self

import pytest
from ddgs.exceptions import DDGSException
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from llm import models


def _fake_long_context_model() -> FakeListChatModel:
    return FakeListChatModel(responses=['summary'])


_original_long_context_model = models.get_long_context_model
models.get_long_context_model = _fake_long_context_model
try:
    from llm.tools import web_search
finally:
    models.get_long_context_model = _original_long_context_model


class FakeDDGS:
    calls: ClassVar[list[dict[str, object]]] = []
    failures: ClassVar[int] = 0

    def __init__(self, **kwargs: object) -> None:
        self.init_kwargs = kwargs

    def __enter__(self) -> Self:
        """Enter the fake search client context."""
        return self

    def __exit__(self, *args: object) -> None:
        """Exit the fake search client context."""

    def text(self, query: str, **kwargs: object) -> list[dict[str, str]]:
        self.calls.append({'query': query, **kwargs})
        if self.failures:
            type(self).failures -= 1
            message = 'backend response changed'
            raise DDGSException(message)
        return [{'title': 'Example', 'href': 'https://example.com', 'body': 'Result body'}]


def _invoke_search(query: str) -> str:
    return asyncio.run(web_search.web_searcher.ainvoke({'query': query}))


def _invoke_reader(url: str = 'https://example.com', query: str | None = None) -> str:
    return asyncio.run(web_search.web_reader.ainvoke({'url': url, 'query': query}))


class FakeAsyncWebCrawler:
    result: ClassVar[SimpleNamespace]

    def __init__(self, **kwargs: object) -> None:
        self.init_kwargs = kwargs

    async def __aenter__(self) -> Self:
        """Enter the fake crawler context."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit the fake crawler context."""

    async def arun(self, **kwargs: object) -> SimpleNamespace:
        return self.result


def test_web_searcher_uses_valid_region(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeDDGS.calls = []
    FakeDDGS.failures = 0
    monkeypatch.setattr(web_search, 'DDGS', FakeDDGS)

    result = _invoke_search('test query')

    assert FakeDDGS.calls == [
        {'query': 'test query', 'region': 'us-en', 'max_results': 10},
    ]
    assert '<title>Example</title>' in result


def test_web_searcher_returns_tool_friendly_error_when_backend_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeDDGS.calls = []
    FakeDDGS.failures = 1
    monkeypatch.setattr(web_search, 'DDGS', FakeDDGS)

    result = _invoke_search('test query')

    assert result.startswith('<search_error>')
    assert 'temporarily unavailable' in result


def test_web_reader_returns_error_when_markdown_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeAsyncWebCrawler.result = SimpleNamespace(
        success=True,
        markdown=None,
        status_code=200,
        error_message=None,
    )
    monkeypatch.setattr(web_search, 'AsyncWebCrawler', FakeAsyncWebCrawler)

    result = _invoke_reader(query='installation')

    assert result.startswith('<read_error>')
    assert 'no readable content' in result


def test_web_reader_handles_empty_bm25_result(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeAsyncWebCrawler.result = SimpleNamespace(
        success=True,
        markdown=SimpleNamespace(fit_markdown=None, raw_markdown='Full page'),
        status_code=200,
        error_message=None,
    )
    monkeypatch.setattr(web_search, 'AsyncWebCrawler', FakeAsyncWebCrawler)

    result = _invoke_reader(query='missing topic')

    assert 'No content matching the query' in result
    assert 'Retry without a query' in result


def test_web_reader_returns_crawl_failure_as_tool_error(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeAsyncWebCrawler.result = SimpleNamespace(
        success=False,
        markdown=None,
        status_code=403,
        error_message='Forbidden',
    )
    monkeypatch.setattr(web_search, 'AsyncWebCrawler', FakeAsyncWebCrawler)

    result = _invoke_reader()

    assert result.startswith('<read_error>')
    assert 'HTTP 403' in result
