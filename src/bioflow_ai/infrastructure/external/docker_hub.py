import json
from functools import lru_cache

import aiohttp
import requests

from bioflow_ai.core.config import get_settings


@lru_cache
def get_access_token(user: str, secret: str) -> str:
    url = 'https://hub.docker.com/v2/auth/token'
    payload = json.dumps({'identifier': user, 'secret': secret})
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}

    proxies = None
    if get_settings().network.USE_PROXY:
        proxies = {'http': get_settings().network.get_proxy_url(), 'https': get_settings().network.get_proxy_url()}

    response = requests.post(url, headers=headers, data=payload, proxies=proxies)
    assert response.status_code == 200
    data = response.json()
    return data['access_token']


def check_repository_tags(namespace: str, repo_name: str) -> int:
    url = f'https://hub.docker.com/v2/namespaces/{namespace}/repositories/{repo_name}/tags'

    user_token = get_access_token(get_settings().docker.USERNAME, get_settings().docker.PASSWORD.get_secret_value())
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {user_token}'}

    proxies = None
    if get_settings().network.USE_PROXY:
        proxies = {'http': get_settings().network.get_proxy_url(), 'https': get_settings().network.get_proxy_url()}

    response = requests.head(url, headers=headers, proxies=proxies)
    return response.status_code


async def acheck_repository_tags(namespace: str, repo_name: str) -> int:
    url = f'https://hub.docker.com/v2/namespaces/{namespace}/repositories/{repo_name}/tags'

    user_token = get_access_token(get_settings().docker.USERNAME, get_settings().docker.PASSWORD.get_secret_value())
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {user_token}'}

    connector = None
    if get_settings().network.USE_PROXY:
        connector = aiohttp.TCPConnector()

    async with aiohttp.ClientSession(connector=connector) as session:
        request_kwargs = {'url': url, 'headers': headers}

        if get_settings().network.USE_PROXY:
            request_kwargs['proxy'] = get_settings().network.get_proxy_url()

        async with session.head(**request_kwargs) as response:
            return response.status


def get_repository_tags(namespace: str, repo_name: str) -> list[str]:
    url = f'https://hub.docker.com/v2/namespaces/{namespace}/repositories/{repo_name}/tags?page=1&page_size=10'

    user_token = get_access_token(get_settings().docker.USERNAME, get_settings().docker.PASSWORD.get_secret_value())
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {user_token}'}

    proxies = None
    if get_settings().network.USE_PROXY:
        proxies = {'http': get_settings().network.get_proxy_url(), 'https': get_settings().network.get_proxy_url()}

    response = requests.get(url, headers=headers, proxies=proxies)
    assert response.status_code == 200
    response_data = response.json()

    tags = [image['name'] for image in response_data['results']]
    return sorted(tags, reverse=True)


async def aget_repository_tags(namespace: str, repo_name: str) -> list[str]:
    url = f'https://hub.docker.com/v2/namespaces/{namespace}/repositories/{repo_name}/tags?page=1&page_size=5'

    user_token = get_access_token(get_settings().docker.USERNAME, get_settings().docker.PASSWORD.get_secret_value())
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {user_token}'}

    connector = None
    if get_settings().network.USE_PROXY:
        connector = aiohttp.TCPConnector()

    async with aiohttp.ClientSession(connector=connector) as session:
        request_kwargs = {'url': url, 'headers': headers}

        if get_settings().network.USE_PROXY:
            request_kwargs['proxy'] = get_settings().network.get_proxy_url()

        async with session.get(**request_kwargs) as response:
            assert response.status == 200
            response_data = await response.json()

            tags = [image['name'] for image in response_data['results']]
            return sorted(tags, reverse=True)
