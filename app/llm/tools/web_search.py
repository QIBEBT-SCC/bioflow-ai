from typing import Any, Optional

from google.ai.generativelanguage_v1beta import Tool as GenAITool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool, ArgsSchema
from pydantic import BaseModel, Field

from app.core.llm_chat import gemini_25_pro, gemini_25_flash


class GoogleSearchInput(BaseModel):
    query: str = Field(description='Keywords for searching')


class GoogleSearch(BaseTool):
    name: str = 'google_search'
    description: str = 'Search information from google'
    args_schema: Optional[ArgsSchema] = GoogleSearchInput
    return_direct: bool = False

    use_pro: bool = False

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        if self.use_pro:
            llm = gemini_25_pro()
        else:
            llm = gemini_25_flash()

        response = llm.invoke(query, tools=[GenAITool(google_search={})])
        return response.content

    async def _arun(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> Any:
        if self.use_pro:
            llm = gemini_25_pro()
        else:
            llm = gemini_25_flash()

        response = await llm.ainvoke(query, tools=[GenAITool(google_search={})])
        return response.content


google_search = GoogleSearch()
google_search_pro = GoogleSearch(use_pro=True)
