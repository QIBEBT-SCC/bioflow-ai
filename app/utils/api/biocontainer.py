from datetime import datetime
from typing import Optional

import aiohttp
import requests
import rich
from pydantic import BaseModel

from app.core.config import get_settings


class BioContainerToolVersion(BaseModel):
    id: str
    meta_version: str
    name: str
    url: str


class BioContainerTool(BaseModel):
    id: str
    description: Optional[str] = None
    pulls: int
    license: Optional[str] = None
    tool_url: Optional[str] = None


class BioContainerImage(BaseModel):
    image_name: str
    image_type: str
    registry_host: str
    updated: Optional[datetime] = None


class BioContainerVersion(BioContainerToolVersion):
    images: list[BioContainerImage]


def search_bio_container(tool_name: str) -> list[BioContainerTool]:
    url = f'https://api.biocontainers.pro/ga4gh/trs/v2/tools?toolname={tool_name}&limit=20&sort_field=pulls&sort_order=desc'
    headers = {'Accept': 'application/json'}

    proxies = None
    if get_settings().network.USE_PROXY:
        proxies = {'http': get_settings().network.get_proxy_url(), 'https': get_settings().network.get_proxy_url()}

    response = requests.get(url, headers=headers, proxies=proxies)
    assert response.status_code == 200
    response_data = response.json()

    return [BioContainerTool.model_validate(r) for r in response_data]


async def asearch_bio_container(tool_name: str) -> list[BioContainerTool]:
    url = f'https://api.biocontainers.pro/ga4gh/trs/v2/tools?toolname={tool_name}&limit=20&sort_field=pulls&sort_order=desc'
    headers = {'Accept': 'application/json'}

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

            return [BioContainerTool.model_validate(r) for r in response_data]


def get_bio_container_image(tool_name: str) -> Optional[str]:
    url = f'https://api.biocontainers.pro/ga4gh/trs/v2/tools/{tool_name}/versions'
    headers = {'Accept': 'application/json'}

    proxies = None
    if get_settings().network.USE_PROXY:
        proxies = {'http': get_settings().network.get_proxy_url(), 'https': get_settings().network.get_proxy_url()}

    response = requests.get(url, headers=headers, proxies=proxies)
    assert response.status_code == 200
    last_version = BioContainerVersion.model_validate(response.json()[-1])
    for image in last_version.images:
        if image.image_type == 'Docker':
            return image.image_name

    return None


async def aget_bio_container_image(tool_name: str) -> Optional[str]:
    url = f'https://api.biocontainers.pro/ga4gh/trs/v2/tools/{tool_name}/versions'
    headers = {'Accept': 'application/json'}

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
            last_version = BioContainerVersion.model_validate(response_data[-1])
            for image in last_version.images:
                if image.image_type == 'Docker':
                    return image.image_name

            return None
