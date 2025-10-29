import json
import uuid
from typing import Any, Dict


def _sse_line(data: Dict[str, Any]) -> str:
    """Serialize a dict as one SSE data line followed by a blank line."""
    return f'data: {json.dumps(data, ensure_ascii=False)}\n\n'


# ============================================================================
# Message Start Part
# ============================================================================


def start_message(message_id: str | None = None) -> str:
    """Start of a UI message with optional explicit id."""
    if message_id is None:
        message_id = str(uuid.uuid4())
    return _sse_line({'type': 'start', 'messageId': message_id})


# ============================================================================
# Text Parts
# ============================================================================


def text_start(text_id: str | None = None) -> str:
    """Indicates the beginning of a text block."""
    if text_id is None:
        text_id = f'msg_{uuid.uuid4().hex}'
    return _sse_line({'type': 'text-start', 'id': text_id})


def text_delta(text_id: str, delta: str) -> str:
    """Contains incremental text content for the text block."""
    return _sse_line({'type': 'text-delta', 'id': text_id, 'delta': delta})


def text_end(text_id: str) -> str:
    """Indicates the completion of a text block."""
    return _sse_line({'type': 'text-end', 'id': text_id})


# ============================================================================
# Reasoning Parts
# ============================================================================


def reasoning_start(reasoning_id: str | None = None) -> str:
    """Indicates the beginning of a reasoning block."""
    if reasoning_id is None:
        reasoning_id = f'reasoning_{uuid.uuid4().hex}'
    return _sse_line({'type': 'reasoning-start', 'id': reasoning_id})


def reasoning_delta(reasoning_id: str, delta: str) -> str:
    """Contains incremental reasoning content for the reasoning block."""
    return _sse_line({'type': 'reasoning-delta', 'id': reasoning_id, 'delta': delta})


def reasoning_end(reasoning_id: str) -> str:
    """Indicates the completion of a reasoning block."""
    return _sse_line({'type': 'reasoning-end', 'id': reasoning_id})


# ============================================================================
# Source Parts
# ============================================================================


def source_url(source_id: str, url: str) -> str:
    """References to external URLs."""
    return _sse_line({'type': 'source-url', 'sourceId': source_id, 'url': url})


def source_document(source_id: str, media_type: str, title: str) -> str:
    """References to documents or files."""
    return _sse_line({'type': 'source-document', 'sourceId': source_id, 'mediaType': media_type, 'title': title})


# ============================================================================
# File Part
# ============================================================================


def file_part(url: str, media_type: str) -> str:
    """The file parts contain references to files with their media type."""
    return _sse_line({'type': 'file', 'url': url, 'mediaType': media_type})


# ============================================================================
# Data Parts
# ============================================================================


def data_part(suffix: str, data: Dict[str, Any]) -> str:
    """Custom data-* part per AI SDK UI protocol."""
    return _sse_line({'type': f'data-{suffix}', 'data': data})


def data_part_typed(data_type: str, data: Dict[str, Any]) -> str:
    """Typed custom data part supporting data-* format."""
    return _sse_line({'type': f'data-{data_type}', 'data': data})


def data_part_raw(type_name: str, **kwargs) -> str:
    """Raw data part with directly specified type name."""
    return _sse_line({'type': type_name, **kwargs})


# ============================================================================
# Error Part
# ============================================================================


def error_part(message: str) -> str:
    """The error parts are appended to the message as they are received."""
    return _sse_line({'type': 'error', 'errorText': message})


# ============================================================================
# Tool Parts
# ============================================================================


def tool_input_start(tool_call_id: str, tool_name: str) -> str:
    """Indicates the beginning of tool input streaming."""
    return _sse_line({'type': 'tool-input-start', 'toolCallId': tool_call_id, 'toolName': tool_name})


def tool_input_delta(tool_call_id: str, input_text_delta: str) -> str:
    """Incremental chunks of tool input as it's being generated."""
    return _sse_line({'type': 'tool-input-delta', 'toolCallId': tool_call_id, 'inputTextDelta': input_text_delta})


def tool_input_available(tool_call_id: str, tool_name: str, input_data: Dict[str, Any]) -> str:
    """Indicates that tool input is complete and ready for execution."""
    return _sse_line(
        {'type': 'tool-input-available', 'toolCallId': tool_call_id, 'toolName': tool_name, 'input': input_data}
    )


def tool_output_available(tool_call_id: str, output_data: Dict[str, Any]) -> str:
    """Contains the result of tool execution."""
    return _sse_line({'type': 'tool-output-available', 'toolCallId': tool_call_id, 'output': output_data})


# ============================================================================
# Start Step Part
# ============================================================================


def start_step() -> str:
    """A part indicating the start of a step."""
    return _sse_line({'type': 'start-step'})


# ============================================================================
# Finish Step Part
# ============================================================================


def finish_step() -> str:
    """A part indicating that a step has been completed."""
    return _sse_line({'type': 'finish-step'})


# ============================================================================
# Finish Message Part
# ============================================================================


def finish_message() -> str:
    """A part indicating the completion of a message."""
    return _sse_line({'type': 'finish'})


# ============================================================================
# Stream Termination
# ============================================================================


def done_marker() -> str:
    """The stream ends with a special [DONE] marker."""
    return 'data: [DONE]\n\n'


# ============================================================================
# Utility Functions
# ============================================================================


def generate_text_id() -> str:
    """Generate text block ID."""
    return f'msg_{uuid.uuid4().hex}'


def generate_reasoning_id() -> str:
    """Generate reasoning block ID."""
    return f'reasoning_{uuid.uuid4().hex}'


def generate_tool_call_id() -> str:
    """Generate tool call ID."""
    return f'call_{uuid.uuid4().hex}'


def generate_source_id() -> str:
    """Generate source ID."""
    return f'source_{uuid.uuid4().hex}'
