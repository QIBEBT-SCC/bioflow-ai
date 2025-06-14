from datetime import datetime
from typing import Optional

import aiohttp
from pydantic import BaseModel

from app.core.config import get_settings


class BioContainerToolVersion(BaseModel):
    id: str
    meta_version: str
    name: str
    url: str


class BioContainerTool(BaseModel):
    id: str
    license: str
    tool_url: str
    versions: list[BioContainerToolVersion]


class BioContainerImage(BaseModel):
    image_name: str
    image_type: str
    registry_host: str
    updated: Optional[datetime] = None


class BioContainerVersion(BioContainerToolVersion):
    images: list[BioContainerImage]


async def search_tool(tool_name: str) -> BioContainerTool:
    url = f'https://api.biocontainers.pro/ga4gh/trs/v2/tools?toolname={tool_name}&limit=20&sort_field=id&sort_order=asc'
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

            return BioContainerTool.model_validate(response_data[0])


async def get_version_info(tool_name: str, version: str):
    url = f'https://api.biocontainers.pro/ga4gh/trs/v2/tools/{tool_name}/versions/{version}'
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

            return BioContainerVersion.model_validate(response_data)
