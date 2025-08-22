import re
from pathlib import Path

from app.core.config import get_settings


def view_testdata() -> str:
    """Reads the test data description file

    Reads the test data description file and returns the content with HTML comments removed.

    Returns:
        The processed test data description text.
    """
    testdata_description_file = Path(get_settings().persistent.TEST_DATA_PATH) / 'DESCRIPTION.md'

    try:
        with open(testdata_description_file, 'r', encoding='utf-8') as f:
            origin_text = f.read()

        cleaned_text = re.sub(r'<!--.*?-->\n', '', origin_text, flags=re.DOTALL)
        return cleaned_text

    except FileNotFoundError:
        return f'Error: File {testdata_description_file} not found'
    except Exception as e:
        return f'Error reading file: {str(e)}'
