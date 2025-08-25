from operator import itemgetter
from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.tools import BaseTool, ArgsSchema, ToolException
from pydantic import BaseModel, Field

from app.core.llm_chat import gpt_41_mini
from app.models.tools import ImageConfig
from app.utils.api.biocontainer import (
    search_bio_container,
    BioContainerTool,
    get_bio_container_image,
    asearch_bio_container,
    aget_bio_container_image,
)
from app.utils.api.docker_hub import (
    check_repository_tags,
    get_repository_tags,
    acheck_repository_tags,
    aget_repository_tags,
)

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


class SelectResult(BaseModel):
    correct_image_id: int = Field(description='The <id/> value of the correct image, or -1 if it is not available')


class SearchImageInput(BaseModel):
    tool_name: str = Field(description='Possible tool docker image names, usually lowercase letters')
    origin_text: str = Field(description='origin user require')


class SearchImage(BaseTool):
    name: str = 'search_image'
    description: str = 'Search for image of a tool from the appropriate registries'
    args_schema: Optional[ArgsSchema] = SearchImageInput
    return_direct: bool = False

    def _run(
        self,
        tool_name: str,
        origin_text: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) ->str:
        # 从项目专有镜像库中搜索
        if check_repository_tags('aye1032', tool_name.lower()) == 200:
            tags = get_repository_tags('aye1032', tool_name.lower())
            if (tag := tags.pop(0)) == 'latest':
                tag = tags[0]
            image = ImageConfig(namespace='aye1032', repository=tool_name.lower(), tag=tag)
            return f'Found pullable tool docker image: `{image.get_image()}`'

        # # 从staphb仓库中搜索
        # if check_repository_tags('staphb', tool_name.lower()) == 200:
        #     tags = get_repository_tags('staphb', tool_name.lower())
        #     if (tag := tags.pop(0)) == 'latest':
        #         tag = tags[0]
        #     return ImageConfig(namespace='staphb', repository=tool_name.lower(), tag=tag)

        # 从 BioContainer 中搜索
        else:
            bio_container_tools = search_bio_container(tool_name.lower())
            if len(bio_container_tools) == 0:
                raise ToolException('No image for this tool was found in the docker repository')

            llm = gpt_41_mini()
            formatter = itemgetter('bio_container_tools') | RunnableLambda(self.image_formatter)
            prompt = ChatPromptTemplate(messages=[SystemMessage(content=SELECT_IMAGE_SYS), ('human', SELECT_HUMAN)])
            chain = RunnablePassthrough.assign(tool_info=formatter) | prompt | llm.with_structured_output(SelectResult)

            select_result: SelectResult = chain.invoke(
                {'bio_container_tools': bio_container_tools, 'origin_text': origin_text}
            )

            if select_result.correct_image_id == -1:
                raise ToolException('No image for this tool was found in the docker repository')

            selected_image = bio_container_tools[select_result.correct_image_id].id
            latest_image = get_bio_container_image(selected_image)
            return f'Found pullable tool docker image: `{latest_image}`'

    async def _arun(
        self,
        tool_name: str,
        origin_text: str,
        tool_call_id: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    )->str:
        # 从项目专有镜像库中搜索
        status_code = await acheck_repository_tags('aye1032', tool_name.lower())
        if status_code == 200:
            tags = await aget_repository_tags('aye1032', tool_name.lower())
            if (tag := tags.pop(0)) == 'latest':
                tag = tag[0]
            image = ImageConfig(namespace='aye1032', repository=tool_name.lower(), tag=tag)
            return f'Found pullable tool docker image: `{image.get_image()}`'

        # 从 BioContainer 中搜索
        else:
            bio_container_tools = await asearch_bio_container(tool_name.lower())
            if len(bio_container_tools) == 0:
                raise ToolException('No image for this tool was found in the docker repository')

            llm = gpt_41_mini()
            formatter = itemgetter('bio_container_tools') | RunnableLambda(self.image_formatter)
            prompt = ChatPromptTemplate(messages=[SystemMessage(content=SELECT_IMAGE_SYS), ('human', SELECT_HUMAN)])
            chain = RunnablePassthrough.assign(tool_info=formatter) | prompt | llm.with_structured_output(SelectResult)

            select_result: SelectResult = await chain.ainvoke(
                {'bio_container_tools': bio_container_tools, 'origin_text': origin_text}
            )

            if select_result.correct_image_id == -1:
                raise ToolException('No image for this tool was found in the docker repository')

            selected_image = bio_container_tools[select_result.correct_image_id].id
            latest_image = await aget_bio_container_image(selected_image)
            return f'Found pullable tool docker image: `{latest_image}`'

    @staticmethod
    def image_formatter(tools: list[BioContainerTool]) -> str:
        formatted = [
            (
                '   <image>\n'
                f'      <id>{index}</id>\n'
                f'      <image_name>{t.id}</image_name>\n'
                f'      <description>{t.description}</description>\n'
                f'      <pull_count>{t.pulls}</pull_count>\n'
                f'      <tool_url>{t.tool_url}</tool_url>\n'
                '   </image>'
            )
            for index, t in enumerate(tools)
        ]

        return '\n\n'.join(formatted)


search_image = SearchImage()
