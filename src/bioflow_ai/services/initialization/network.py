"""Network and proxy connectivity preflight checks for auto-init."""

from __future__ import annotations

import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict

from bioflow_ai.services.initialization.common import detect_shell_proxy

if TYPE_CHECKING:
    from collections.abc import Sequence

    from bioflow_ai.core.config import NetworkConfig

DEFAULT_PROXY_TIMEOUT = 3.0
DEFAULT_ENDPOINT_TIMEOUT = 8.0

_MS_PER_SECOND = 1000
_REGISTRY_ACCEPTED_STATUSES = frozenset({HTTPStatus.OK, HTTPStatus.UNAUTHORIZED})
_HF_ACCEPTED_STATUSES = frozenset(
    {
        HTTPStatus.OK,
        HTTPStatus.MOVED_PERMANENTLY,
        HTTPStatus.FOUND,
        HTTPStatus.TEMPORARY_REDIRECT,
        HTTPStatus.PERMANENT_REDIRECT,
        HTTPStatus.UNAUTHORIZED,
    }
)


class PreflightTarget(BaseModel):
    """One external HTTP/HTTPS target checked during network preflight."""

    model_config = ConfigDict(frozen=True)

    key: str
    name: str
    domain: str
    url: str
    purpose: str
    method: Literal['GET', 'HEAD'] = 'GET'
    accepted_statuses: frozenset[int]


class ProxyCheckResult(BaseModel):
    """TCP reachability check result for a configured network proxy."""

    model_config = ConfigDict(frozen=True)

    enabled: bool
    proxy_url: str | None = None
    reachable: bool = True
    latency_ms: int | None = None
    error: str | None = None


class EndpointCheckResult(BaseModel):
    """HTTP/HTTPS connectivity check result for one external dependency."""

    model_config = ConfigDict(frozen=True)

    name: str
    domain: str
    url: str
    purpose: str
    ok: bool
    status_code: int | None = None
    latency_ms: int | None = None
    error: str | None = None


class NetworkPreflightResult(BaseModel):
    """Aggregated proxy and external endpoint check outcomes."""

    model_config = ConfigDict(frozen=True)

    proxy: ProxyCheckResult
    endpoints: list[EndpointCheckResult]
    shell_proxy_detected: str | None = None

    @property
    def ok(self) -> bool:
        """Whether the proxy (if enabled) and all required endpoints are reachable."""
        return self.proxy.reachable and all(item.ok for item in self.endpoints)

    @property
    def failed_endpoints(self) -> list[EndpointCheckResult]:
        """Return only endpoints that failed the connectivity probe."""
        return [item for item in self.endpoints if not item.ok]


def build_preflight_targets() -> tuple[PreflightTarget, ...]:
    """Build the ordered list of external domains required during initialization."""
    hf_endpoint = (
        os.environ.get('HF_ENDPOINT', 'https://huggingface.co').strip().rstrip('/') or 'https://huggingface.co'
    )
    hf_domain = urlparse(hf_endpoint).netloc or 'huggingface.co'
    return (
        PreflightTarget(
            key='docker_hub',
            name='Docker Hub',
            domain='registry-1.docker.io',
            url='https://registry-1.docker.io/v2/',
            purpose='Base & tool container images',
            method='GET',
            accepted_statuses=_REGISTRY_ACCEPTED_STATUSES,
        ),
        PreflightTarget(
            key='ghcr',
            name='GHCR',
            domain='ghcr.io',
            url='https://ghcr.io/v2/',
            purpose='Web UI container image',
            method='GET',
            accepted_statuses=_REGISTRY_ACCEPTED_STATUSES,
        ),
        PreflightTarget(
            key='quay',
            name='Quay.io',
            domain='quay.io',
            url='https://quay.io/v2/',
            purpose='etcd & bioinformatics tool images',
            method='GET',
            accepted_statuses=_REGISTRY_ACCEPTED_STATUSES,
        ),
        PreflightTarget(
            key='huggingface',
            name='Hugging Face',
            domain=hf_domain,
            url=f'{hf_endpoint}/',
            purpose='Preset database & test dataset',
            method='HEAD',
            accepted_statuses=_HF_ACCEPTED_STATUSES,
        ),
    )


def check_proxy_tcp(network: NetworkConfig, *, timeout: float = DEFAULT_PROXY_TIMEOUT) -> ProxyCheckResult:
    """Verify TCP connectivity to the configured proxy host and port."""
    if not network.USE_PROXY:
        return ProxyCheckResult(enabled=False, reachable=True)

    proxy_url = network.get_proxy_url()
    start = time.monotonic()
    try:
        with socket.create_connection((network.PROXY_HOST, network.PROXY_PORT), timeout=timeout):
            pass
    except OSError as exc:
        latency_ms = max(1, round((time.monotonic() - start) * _MS_PER_SECOND))
        return ProxyCheckResult(
            enabled=True,
            proxy_url=proxy_url,
            reachable=False,
            latency_ms=latency_ms,
            error=str(exc) or type(exc).__name__,
        )

    latency_ms = max(1, round((time.monotonic() - start) * _MS_PER_SECOND))
    return ProxyCheckResult(
        enabled=True,
        proxy_url=proxy_url,
        reachable=True,
        latency_ms=latency_ms,
    )


def _probe_endpoint(
    target: PreflightTarget,
    *,
    proxy_url: str | None,
    timeout: float,
) -> EndpointCheckResult:
    start = time.monotonic()
    try:
        with httpx.Client(
            proxy=proxy_url,
            trust_env=False,
            timeout=timeout,
            follow_redirects=False,
        ) as client:
            with client.stream(target.method, target.url) as response:
                status_code = response.status_code
            if status_code == HTTPStatus.METHOD_NOT_ALLOWED and target.method == 'HEAD':
                with client.stream('GET', target.url) as response:
                    status_code = response.status_code
    except (httpx.ProxyError, httpx.TimeoutException, httpx.HTTPError, OSError) as exc:
        latency_ms = max(1, round((time.monotonic() - start) * _MS_PER_SECOND))
        return EndpointCheckResult(
            name=target.name,
            domain=target.domain,
            url=target.url,
            purpose=target.purpose,
            ok=False,
            latency_ms=latency_ms,
            error=str(exc) or type(exc).__name__,
        )

    latency_ms = max(1, round((time.monotonic() - start) * _MS_PER_SECOND))
    ok = status_code in target.accepted_statuses
    return EndpointCheckResult(
        name=target.name,
        domain=target.domain,
        url=target.url,
        purpose=target.purpose,
        ok=ok,
        status_code=status_code,
        latency_ms=latency_ms,
        error=None if ok else f'Unexpected HTTP {status_code}',
    )


def run_network_preflight(
    network: NetworkConfig,
    *,
    proxy_timeout: float = DEFAULT_PROXY_TIMEOUT,
    endpoint_timeout: float = DEFAULT_ENDPOINT_TIMEOUT,
    targets: Sequence[PreflightTarget] | None = None,
) -> NetworkPreflightResult:
    """Check proxy reachability (when enabled) and probe all required external domains."""
    proxy = check_proxy_tcp(network, timeout=proxy_timeout)
    shell_proxy_detected = None if network.USE_PROXY else detect_shell_proxy()

    if network.USE_PROXY and not proxy.reachable:
        return NetworkPreflightResult(
            proxy=proxy,
            endpoints=[],
            shell_proxy_detected=None,
        )

    resolved_targets = tuple(targets) if targets is not None else build_preflight_targets()
    proxy_url = network.get_proxy_url() if network.USE_PROXY else None
    with ThreadPoolExecutor(max_workers=max(1, len(resolved_targets))) as executor:
        endpoints = list(
            executor.map(
                lambda target: _probe_endpoint(target, proxy_url=proxy_url, timeout=endpoint_timeout),
                resolved_targets,
            )
        )

    return NetworkPreflightResult(
        proxy=proxy,
        endpoints=endpoints,
        shell_proxy_detected=shell_proxy_detected,
    )
