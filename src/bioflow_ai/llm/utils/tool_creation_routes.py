from typing import Literal

from bioflow_ai.llm.states import CreateToolState

PostReportRoute = Literal['__end__'] | list[Literal['choose_group', 'generate_tool_config', 'process_help_doc']]


def route_after_image_selection(
    state: CreateToolState,
    *,
    skip_search: bool,
) -> Literal['search_exist_tool', 'get_report', '__end__']:
    """Choose the next image-selection branch without scheduling work after a handoff."""
    if state.get('return_to_parent'):
        return '__end__'
    return 'get_report' if skip_search else 'search_exist_tool'


def route_after_tool_report(state: CreateToolState) -> PostReportRoute:
    """End on a parent handoff or fan out the independent post-report work."""
    if state.get('return_to_parent'):
        return '__end__'
    return ['choose_group', 'generate_tool_config', 'process_help_doc']
