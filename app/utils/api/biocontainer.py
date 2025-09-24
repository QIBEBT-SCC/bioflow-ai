from datetime import datetime
from typing import Optional, Literal

import httpx
from pydantic import BaseModel

from app.core.config import get_settings


class BioContainerTool(BaseModel):
    id: str
    description: Optional[str] = None
    pulls: int
    license: Optional[str] = None
    tool_url: Optional[str] = None


class BioContainerImage(BaseModel):
    image_name: str
    image_type: Literal['Docker', 'Singularity', 'Conda']
    size: int
    updated: Optional[datetime] = None


class BioContainerVersion(BaseModel):
    name: str
    meta_version: str
    images: list[BioContainerImage]


def search_biocontainer(tool_name: str) -> list[BioContainerTool]:
    url = f'https://api.biocontainers.pro/ga4gh/trs/v2/tools?toolname={tool_name}&limit=20&sort_field=pulls&sort_order=desc'

    proxy = get_settings().network.get_proxy_url() if get_settings().network.USE_PROXY else None
    with httpx.Client(proxy=proxy) as client:
        response = client.get(url)
        response.raise_for_status()
        return [BioContainerTool.model_validate(r) for r in response.json()]


async def asearch_biocontainer(tool_name: str) -> list[BioContainerTool]:
    url = f'https://api.biocontainers.pro/ga4gh/trs/v2/tools?toolname={tool_name}&limit=20&sort_field=pulls&sort_order=desc'

    proxy = get_settings().network.get_proxy_url() if get_settings().network.USE_PROXY else None
    async with httpx.AsyncClient(proxy=proxy) as client:
        response = await client.get(url)
        response.raise_for_status()
        return [BioContainerTool.model_validate(r) for r in response.json()]


def get_biocontainer_image_tags(tool_name: str) -> list[BioContainerVersion]:
    url = f'https://api.biocontainers.pro/ga4gh/trs/v2/tools/{tool_name}/versions'

    proxy = get_settings().network.get_proxy_url() if get_settings().network.USE_PROXY else None
    with httpx.Client(proxy=proxy) as client:
        response = client.get(url)
        response.raise_for_status()

        versions = []
        for r in response.json():
            version = BioContainerVersion.model_validate(r)
            version.images = [img for img in version.images if img.image_type == 'Docker']
            versions.append(version)

        return versions[::-1]


async def aget_biocontainer_image_tags(tool_name: str) -> list[BioContainerVersion]:
    url = f'https://api.biocontainers.pro/ga4gh/trs/v2/tools/{tool_name}/versions'

    proxy = get_settings().network.get_proxy_url() if get_settings().network.USE_PROXY else None
    async with httpx.AsyncClient(proxy=proxy) as client:
        response = await client.get(url)
        response.raise_for_status()

        versions = []
        for r in response.json():
            version = BioContainerVersion.model_validate(r)
            version.images = [img for img in version.images if img.image_type == 'Docker']
            versions.append(version)

        return versions[::-1]
