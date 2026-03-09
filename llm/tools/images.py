from datetime import datetime
from typing import Literal, Optional

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel

from app.core.config import get_settings

SELECT_IMAGE_SYS = """You will receive the following inputs:
1. A list of container image metadata from BioContainer, each including fields such as `image_name`, `description`, `pull_count`, and `tool_url`.
2. The user's original tool requirement (`origin_require`), describing the functionality or purpose of the desired tool.

Your task is to:
* Analyze the user's original requirement and determine whether any of the provided images match the intended purpose.
* A matching image is one whose functionality clearly satisfies the requirement stated in `origin_require`.
* If multiple images are suitable:
    * First, prefer images pointing to the same tool_url as other candidates but with a higher pull_count (i.e., when multiple containers represent the same tool, choose the most popular one).
    * Then, among different tools, choose the one whose description best matches the requirement, using pull_count as a secondary tie-breaker.
* Return the corresponding `<id>` value (an integer) of the selected image.
* If no image matches the requirement, return `-1`.
"""

SELECT_HUMAN = """The origin require is:
> {origin_text}

Below is available images:
<images>
{tool_info}
</images>
"""

K = 1024


class BioContainerTool(BaseModel):
    id: str
    description: Optional[str] = None
    pulls: int


class BioContainerImage(BaseModel):
    image_name: str
    image_type: Literal['Docker', 'Singularity', 'Conda']
    size: int
    updated: Optional[datetime] = None


class BioContainerVersion(BaseModel):
    name: str
    meta_version: str
    images: list[BioContainerImage]


def _format_size(size_bytes: int) -> str:
    """Format raw byte size to human-readable string.

    Examples: 1024 -> "1.00 KB", 123 -> "123 B"
    """
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = float(size_bytes)
    for unit in units:
        if size < K or unit == 'TB':
            return f'{size:.2f} {unit}'
        size /= K
    return ''


@tool(parse_docstring=True)
async def search_image(tool_name: str) -> str:
    """Search BioContainer to get the image of the target tool.

    Args:
        tool_name: The name of the tool to be found
    """
    url = (
        f'https://api.biocontainers.pro/ga4gh/trs/v2/tools?toolname={tool_name}'
        f'&limit=6&sort_field=pulls&sort_order=desc'
    )

    proxy = get_settings().network.get_proxy_url() if get_settings().network.USE_PROXY else None
    async with httpx.AsyncClient(proxy=proxy) as client:
        response = await client.get(url)
        response.raise_for_status()

    # 检查响应是否有内容
    if response.status_code == 204 or not response.content:  # noqa: PLR2004
        return 'No corresponding mirror found, please try asking human to confirm the tool name.'

    tool_list = [BioContainerTool.model_validate(r) for r in response.json()]
    if not tool_list:
        return 'No corresponding mirror found, please try asking human to confirm the tool name.'

    xml_list = [
        (
            f'  <tool>\n'
            f'    <name>{t.id}</name>\n'
            f'    <description>{t.description}</description>\n'
            f'    <pulls>{t.pulls}</pulls>\n'
            f'  </tool>'
        )
        for t in tool_list
    ]

    return f'<tool>\n{"\n".join(xml_list)}\n</tool>\n\nDetermine the image you need based on its name, description, and pull count.'


@tool(parse_docstring=True)
async def view_image_tags(image_name: str) -> str:
    """View tags for a specified image name.

    Args:
        image_name: Image name to be viewed
    """
    url = f'https://api.biocontainers.pro/ga4gh/trs/v2/tools/{image_name}/versions'

    proxy = get_settings().network.get_proxy_url() if get_settings().network.USE_PROXY else None
    async with httpx.AsyncClient(proxy=proxy) as client:
        response = await client.get(url)
        response.raise_for_status()

        # 检查响应是否有内容
        if response.status_code == 204 or not response.content:  # noqa: PLR2004
            return 'No tags found for this image, please confirm the image name.'

        versions = []
        for r in response.json():
            version = BioContainerVersion.model_validate(r)
            version.images = [img for img in version.images if img.image_type == 'Docker']
            versions.append(version)
        versions = versions[::-1][:5]

    if not versions:
        return 'No tags found for this image, please confirm the image name.'

    version_xml_list = []
    for v in versions:
        image_xml_list = [
            (
                f'    <image>\n'
                f'      <name>{img.image_name}</name>\n'
                f'      <size>{_format_size(img.size)}</size>\n'
                f'      <updated>{img.updated.strftime("%Y-%m-%d %H:%M:%S")}</updated>\n'
                f'    </image>'
            )
            for img in v.images
        ]

        version_xml = (
            f'  <version>\n'
            f'    <meta_version>{v.meta_version}</meta_version>\n'
            f'{("\n".join(image_xml_list)) if image_xml_list else "    <image />"}\n'
            f'  </version>'
        )
        version_xml_list.append(version_xml)

    return (
        f'<versions image="{image_name}">\n'
        f'{"\n".join(version_xml_list)}\n'
        f'</versions>\n\n'
        f'Prefer the latest version of the image. If a version has multiple images, first select the smaller version containing only basic tools based on size (some images include databases, which will cause the image size to be significantly larger than the normal version); if they are of similar size, choose the one with a more recent push time.'
    )
