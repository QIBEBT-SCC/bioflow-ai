from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool, ArgsSchema
from pydantic import BaseModel, Field

from app.core.config import get_settings


class JinaSearchInput(BaseModel):
    query: str = Field(description='query for searching')


class JinaSearch(BaseTool):
    name: str = 'web_search'
    description: str = 'Search information from web'
    args_schema: Optional[ArgsSchema] = JinaSearchInput
    return_direct: bool = False

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        import requests

        url = f'https://s.jina.ai/?q={query}&hl=en'
        headers = {'Accept': 'application/json', 'X-Respond-With': 'no-content'}
        if api_key := get_settings().llm.tools.JINA_API:
            headers['Authorization'] = f'Bearer {api_key.get_secret_value()}'

        response = requests.get(url, headers=headers)

        response_block = [
            (
                f'<result>\n'
                f'  <title>{d["title"]}</title>\n'
                f'  <url>{d["url"]}</url>\n'
                f'  <fragment>{d["description"]}</fragment>\n'
                f'</result>'
            )
            for d in response.json()['data']
        ]
        return '\n\n'.join(response_block)

    async def _arun(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        import aiohttp

        url = f'https://s.jina.ai/?q={query}&hl=en'
        headers = {'Accept': 'application/json', 'X-Respond-With': 'no-content'}
        if api_key := get_settings().llm.tools.JINA_API:
            headers['Authorization'] = f'Bearer {api_key.get_secret_value()}'

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                data = await response.json()

                response_block = [
                    (
                        f'<result>\n'
                        f'  <title>{d["title"]}</title>\n'
                        f'  <url>{d["url"]}</url>\n'
                        f'  <fragment>{d["description"]}</fragment>\n'
                        f'</result>'
                    )
                    for d in data['data']
                ]
                return '\n\n'.join(response_block)


class JinaReaderInput(BaseModel):
    url: str = Field(description='Web page link to read')


class JinaReader(BaseTool):
    name: str = 'web_reader'
    description: str = 'Read the detailed content of the specified URL'
    args_schema: Optional[ArgsSchema] = JinaReaderInput
    return_direct: bool = False

    def _run(self, url: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        import requests

        url = f'https://r.jina.ai/{url}'
        headers = {'X-Remove-Selector': 'header, .class, #id', 'X-Retain-Images': 'none'}
        if api_key := get_settings().llm.tools.JINA_API:
            headers['Authorization'] = f'Bearer {api_key.get_secret_value()}'
        if get_settings().network.USE_PROXY:
            headers['X-Proxy-Url'] = get_settings().network.get_proxy_url()

        response = requests.get(url, headers=headers)
        return response.text

    async def _arun(self, url: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        import aiohttp

        url = f'https://r.jina.ai/{url}'
        headers = {'X-Remove-Selector': 'header, .class, #id', 'X-Retain-Images': 'none'}
        if api_key := get_settings().llm.tools.JINA_API:
            headers['Authorization'] = f'Bearer {api_key.get_secret_value()}'
        if get_settings().network.USE_PROXY:
            headers['X-Proxy-Url'] = get_settings().network.get_proxy_url()

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                data = await response.text()

        return data


jina_search = JinaSearch()
jina_reader = JinaReader()
