import uuid
from typing import TYPE_CHECKING, cast

from langchain.agents import AgentState, create_agent

from bioflow_ai.llm.context import BaseContext
from bioflow_ai.llm.middleware import (
    ChatAttachmentMiddleware,
    DisableParallelToolCallsMiddleware,
    ParentHandoffMiddleware,
    TokenStatisticMiddleware,
)
from bioflow_ai.llm.models import get_agent_model
from bioflow_ai.llm.tools.database import search_local_image, select_image
from bioflow_ai.llm.tools.human import ask_human
from bioflow_ai.schemas.tool_config import ImageConfig
from bioflow_ai.utils.llm.uid_mapper import BidirectionalMapper

if TYPE_CHECKING:
    from bioflow_ai.llm.llm_client import LLMMetadata

SEARCH_IMAGE_LLM = get_agent_model()
metadata = cast('LLMMetadata', SEARCH_IMAGE_LLM.metadata)

SEARCH_IMAGE_SYSTEM = """You are a bioinformatics data engineer. Your task is to find the corresponding tool's image based on the user's needs and return it.

## Mandatory Search Policy
Do not treat one unsuccessful `search_local_image` call as proof that an image is unavailable. Before concluding that no
local image exists, you MUST make at least three sequential search attempts with materially different query strategies:
1. Search for the concise canonical tool or suite name (for example, `GATK` rather than a full task description).
2. Search for a different recognized identifier, such as the executable, package/repository name, common alias, or
   expanded name.
3. Search using the relevant subcommand or a broader functional keyword.

Removing or adding punctuation, changing only letter case, or repeating essentially the same phrase does not count as a
different strategy. If a search returns results but none is suitable, that attempt also counts as unsuccessful and you
must continue with the remaining strategies. Stop early only when a suitable image is found.

After a suitable image is found, use `select_image(image_ref=...)` with the exact `<image_ref>` value. You may call
`ask_human` to request that the user add an image only after all three search strategies have failed. Never call
`ask_human` immediately after the first empty or irrelevant result.
"""


class SearchImageState(AgentState, total=False):
    id_mapper: BidirectionalMapper

    image_uid: uuid.UUID
    image: ImageConfig

    name: str
    description: str
    homepage: str
    paper_link: str


# TODO: 目前是手动补充
search_image_agent = create_agent(
    model=get_agent_model(),
    tools=[
        search_local_image,
        select_image,
        ask_human,
    ],
    system_prompt=SEARCH_IMAGE_SYSTEM,
    middleware=[
        DisableParallelToolCallsMiddleware(),
        ParentHandoffMiddleware(),
        ChatAttachmentMiddleware(),
        TokenStatisticMiddleware('search_image', **metadata),
    ],
    state_schema=SearchImageState,
    context_schema=BaseContext,
)
