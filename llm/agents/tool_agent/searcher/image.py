import uuid
from typing import TYPE_CHECKING, cast

from langchain.agents import AgentState, create_agent

from app.models.tools import ImageConfig
from llm.context import BaseContext
from llm.models import get_agent_model
from llm.tools.database import search_local_image, select_image
from llm.tools.human import ask_human
from llm.utils.middleware import DisableParallelToolCallsMiddleware, TokenStatisticMiddleware
from llm.utils.uid_mapper import BidirectionalMapper

if TYPE_CHECKING:
    from llm.llm_client import LLMMetadata

SEARCH_IMAGE_LLM = get_agent_model()
metadata = cast('LLMMetadata', SEARCH_IMAGE_LLM.metadata)

SEARCH_IMAGE_SYSTEM = """You are a bioinformatics data engineer. Your task is to find the corresponding tool's image based on the user's needs and return it.

According to the tool search results, if there is an image of the tool required by the user, use `select_image` with the exact image URL shown in the `<url>` field to select it;
Otherwise, call the `ask_human` tool to request the user to add an available image.
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
# pyrefly: ignore [no-matching-overload]
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
        TokenStatisticMiddleware('search_image', **metadata),
    ],
    state_schema=SearchImageState,
    context_schema=BaseContext,
)
