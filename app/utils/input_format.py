import os
from os import PathLike
from pathlib import Path
from typing import Optional


def get_path(
    r1: str | PathLike, r2: Optional[str | PathLike] = None, *, output_dir: str | PathLike
) -> tuple[Path, Path, Path, str]:
    """将输入的双端文件转换为pathlib格式

    Args:
        r1: pair-end 文件 1
        r2: pair-end 文件 2
        output_dir: 软件输出目录
    """
    r1_file = Path(r1) if isinstance(r1, str) else r1
    r2_file = Path(r2) if isinstance(r2, str) else r2
    output_path = Path(output_dir) if isinstance(output_dir, str) else output_dir

    sample_name = os.path.commonprefix([r1_file.name, r2_file.name]).strip('_')

    return r1_file, r2_file, output_path, sample_name
