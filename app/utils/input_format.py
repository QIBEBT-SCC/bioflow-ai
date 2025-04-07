import os
from pathlib import Path


def get_path(
    r1: str | Path, r2: str | Path, *, output_dir: str | Path
) -> tuple[Path, Path, Path, str]:
    """将输入的双端文件转换为pathlib格式

    Args:
        r1: pair-end 文件 1
        r2: pair-end 文件 2
        output_dir: 软件输出目录
    """
    r1_file = r1 if isinstance(r1, Path) else Path(r1)
    r2_file = r2 if isinstance(r2, Path) else Path(r2)
    output_path = output_dir if isinstance(output_dir, Path) else Path(output_dir)

    sample_name = os.path.commonprefix([r1_file.name, r2_file.name]).strip('_R').strip('_')

    output_path.mkdir(parents=True, exist_ok=True)

    return r1_file, r2_file, output_path, sample_name
