import json
from functools import lru_cache

import aiohttp
import requests

from bioflow_ai.core.config import get_settings

REQUEST_TIMEOUT = 30


@lru_cache
def get_access_token(user: str, secret: str) -> str:
    """Fetch a Docker Hub access token for the given credentials."""
    url = 'https://hub.docker.com/v2/auth/token'
    payload = json.dumps({'identifier': user, 'secret': secret})
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}

    proxies = None
    if get_settings().network.USE_PROXY:
        proxies = {'http': get_settings().network.get_proxy_url(), 'https': get_settings().network.get_proxy_url()}

    response = requests.post(url, headers=headers, data=payload, proxies=proxies, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    return data['access_token']


def check_repository_tags(namespace: str, repo_name: str) -> int:
    """Return the HTTP status code when checking Docker Hub repository tags."""
    url = f'https://hub.docker.com/v2/namespaces/{namespace}/repositories/{repo_name}/tags'

    user_token = get_access_token(get_settings().docker.USERNAME, get_settings().docker.PASSWORD.get_secret_value())
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {user_token}'}

    proxies = None
    if get_settings().network.USE_PROXY:
        proxies = {'http': get_settings().network.get_proxy_url(), 'https': get_settings().network.get_proxy_url()}

    response = requests.head(url, headers=headers, proxies=proxies, timeout=REQUEST_TIMEOUT)
    return response.status_code


async def acheck_repository_tags(namespace: str, repo_name: str) -> int:
    """Asynchronously return the HTTP status code for Docker Hub repository tags."""
    url = f'https://hub.docker.com/v2/namespaces/{namespace}/repositories/{repo_name}/tags'

    user_token = get_access_token(get_settings().docker.USERNAME, get_settings().docker.PASSWORD.get_secret_value())
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {user_token}'}

    connector = None
    if get_settings().network.USE_PROXY:
        connector = aiohttp.TCPConnector()

    proxy = get_settings().network.get_proxy_url() if get_settings().network.USE_PROXY else None
    async with (
        aiohttp.ClientSession(connector=connector) as session,
        session.head(url, headers=headers, proxy=proxy) as response,
    ):
        return response.status


def get_repository_tags(namespace: str, repo_name: str) -> list[str]:
    """Fetch and sort the latest Docker Hub tags for a repository."""
    url = f'https://hub.docker.com/v2/namespaces/{namespace}/repositories/{repo_name}/tags?page=1&page_size=10'

    user_token = get_access_token(get_settings().docker.USERNAME, get_settings().docker.PASSWORD.get_secret_value())
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {user_token}'}

    proxies = None
    if get_settings().network.USE_PROXY:
        proxies = {'http': get_settings().network.get_proxy_url(), 'https': get_settings().network.get_proxy_url()}

    response = requests.get(url, headers=headers, proxies=proxies, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    response_data = response.json()

    tags = [image['name'] for image in response_data['results']]
    return sorted(tags, reverse=True)


async def aget_repository_tags(namespace: str, repo_name: str) -> list[str]:
    """Asynchronously fetch and sort the latest Docker Hub tags for a repository."""
    url = f'https://hub.docker.com/v2/namespaces/{namespace}/repositories/{repo_name}/tags?page=1&page_size=5'

    user_token = get_access_token(get_settings().docker.USERNAME, get_settings().docker.PASSWORD.get_secret_value())
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {user_token}'}

    connector = None
    if get_settings().network.USE_PROXY:
        connector = aiohttp.TCPConnector()

    proxy = get_settings().network.get_proxy_url() if get_settings().network.USE_PROXY else None
    async with (
        aiohttp.ClientSession(connector=connector) as session,
        session.get(url, headers=headers, proxy=proxy) as response,
    ):
        response.raise_for_status()
        response_data = await response.json()

        tags = [image['name'] for image in response_data['results']]
        return sorted(tags, reverse=True)
