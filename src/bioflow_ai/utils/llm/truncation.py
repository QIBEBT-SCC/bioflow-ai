"""Token-budgeted text previews for language model tool output."""

APPROX_BYTES_PER_TOKEN = 4
DEFAULT_MAX_OUTPUT_TOKENS = 3000


def truncate_output(output: str, *, max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS) -> str:
    """Keep the head and tail of output within an approximate token budget.

    Retain text in a 3:1 head-to-tail ratio, with the notice outside the content
    budget. Estimate one token per four UTF-8 bytes and preserve character boundaries.
    """
    if max_tokens <= 0:
        msg = 'max_tokens must be positive.'
        raise ValueError(msg)

    output_bytes = output.encode('utf-8')
    original_tokens = (len(output_bytes) + APPROX_BYTES_PER_TOKEN - 1) // APPROX_BYTES_PER_TOKEN
    max_bytes = max_tokens * APPROX_BYTES_PER_TOKEN
    if original_tokens <= max_tokens:
        return output

    head_bytes = max_bytes * 3 // 4
    tail_bytes = max_bytes - head_bytes
    head = output_bytes[:head_bytes].decode('utf-8', errors='ignore')
    tail = output_bytes[-tail_bytes:].decode('utf-8', errors='ignore') if tail_bytes else ''
    return f'{head}\n…{original_tokens - max_tokens} tokens truncated…\n{tail}'
