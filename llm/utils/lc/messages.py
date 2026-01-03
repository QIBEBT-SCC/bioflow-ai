from typing import Any, Literal, Union

from langchain_core.messages import BaseMessage


class LinkMessage(BaseMessage):
    link: str = ''
    type: Literal['link'] = 'link'

    def __init__(self, content: Union[str, list[Union[str, dict]]], **kwargs: Any) -> None:
        """Pass in content as positional arg.

        Args:
            content: The string contents of the message.
            kwargs: Additional fields to pass to the message.
        """
        super().__init__(content=content, **kwargs)


class InterruptMessage(BaseMessage):
    type: Literal['interrupt'] = 'interrupt'

    status: Literal['waiting', 'approved', 'rejected'] = 'waiting'

    def __init__(self, content: Union[str, list[Union[str, dict]]], **kwargs: Any) -> None:
        """Pass in content as positional arg.

        Args:
            content: The string contents of the message.
            kwargs: Additional fields to pass to the message.
        """
        super().__init__(content=content, **kwargs)
