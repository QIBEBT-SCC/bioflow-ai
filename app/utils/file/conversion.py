import base64
import mimetypes
from pathlib import Path
from urllib import request


def file_to_data_url(file_path: str) -> str:
    """Convert a local file into a base64 data URL string.

    Args:
        file_path: Absolute or relative path to the file that should be encoded.

    Returns:
        A data URL (e.g. ``data:<mime>;base64,<payload>``) representing the file's content.

    Raises:
        FileNotFoundError: If ``file_path`` does not exist on disk.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f'文件未找到: {file_path}')

    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'

    with open(path, 'rb') as f:
        file_bytes = f.read()
        base64_str = base64.b64encode(file_bytes).decode('utf-8')

    return f'data:{mime_type};base64,{base64_str}'


def data_url_to_file(data_url: str, output_path: str) -> str:
    """Download/parse a data URL and persist it to ``output_path``.

    Args:
        data_url: A ``data:`` URL or HTTP(S) URL whose content will be written to disk.
        output_path: Destination path (file or filename) where the decoded bytes are saved.

    Returns:
        The resolved path to the written file for downstream use.
    """
    with request.urlopen(data_url) as response:
        data = response.read()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return str(out)
