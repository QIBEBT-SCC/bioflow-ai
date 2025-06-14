from functools import wraps
from typing import Any

import aiohttp
import json

from app.core.config import get_settings


def async_lru_cache(maxsize=128):
    """异步版本的LRU缓存装饰器"""
    def decorator(func):
        cache = {}
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = str(args) + str(sorted(kwargs.items()))
            if key in cache:
                return cache[key]
            result = await func(*args, **kwargs)
            if len(cache) >= maxsize:
                cache.clear()
            cache[key] = result
            return result
        return wrapper
    return decorator


@async_lru_cache(maxsize=128)
async def get_access_token(user: str, secret: str) -> str:
    url = 'https://hub.docker.com/v2/auth/token'
    payload = json.dumps({'identifier': user, 'secret': secret})
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    
    connector = None
    if get_settings().network.USE_PROXY:
        connector = aiohttp.TCPConnector()
    
    async with aiohttp.ClientSession(connector=connector) as session:
        request_kwargs = {
            'url': url,
            'headers': headers,
            'data': payload
        }
        
        if get_settings().network.USE_PROXY:
            request_kwargs['proxy'] = get_settings().network.get_proxy_url()
            
        async with session.post(**request_kwargs) as response:
            assert response.status == 200
            data = await response.json()
            return data['access_token']


async def check_repository_tags(namespace: str, repo_name: str) -> dict[str, Any]:
    url = f'https://hub.docker.com/v2/namespaces/{namespace}/repositories/{repo_name}/tags'
    
    user_token = await get_access_token(get_settings().docker.USERNAME, get_settings().docker.PASSWORD.get_secret_value())
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {user_token}'}
    
    connector = None
    if get_settings().network.USE_PROXY:
        connector = aiohttp.TCPConnector()
    
    async with aiohttp.ClientSession(connector=connector) as session:
        request_kwargs = {
            'url': url,
            'headers': headers
        }
        
        if get_settings().network.USE_PROXY:
            request_kwargs['proxy'] = get_settings().network.get_proxy_url()
            
        async with session.head(**request_kwargs) as response:
            assert response.status == 200
            return dict(response.headers)


async def get_repository_tags(namespace: str, repo_name: str) -> list[str]:
    url = f'https://hub.docker.com/v2/namespaces/{namespace}/repositories/{repo_name}/tags?page=1&page_size=5'
    
    user_token = await get_access_token(get_settings().docker.USERNAME, get_settings().docker.PASSWORD.get_secret_value())
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {user_token}'}
    
    connector = None
    if get_settings().network.USE_PROXY:
        connector = aiohttp.TCPConnector()
    
    async with aiohttp.ClientSession(connector=connector) as session:
        request_kwargs = {
            'url': url,
            'headers': headers
        }
        
        if get_settings().network.USE_PROXY:
            request_kwargs['proxy'] = get_settings().network.get_proxy_url()
            
        async with session.get(**request_kwargs) as response:
            assert response.status == 200
            response_data = await response.json()
            
            return [image['name'] for image in response_data['results']]
