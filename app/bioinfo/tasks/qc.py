from os import PathLike
from pathlib import Path
from typing import Optional

import docker
from prefect import task
from pydantic import BaseModel

from app.bioinfo.tasks import FASTP_IMAGE
from app.utils.input_format import get_path


class FastpOutput(BaseModel):
    r1: Path
    r2: Optional[Path] = None
    json_report: Path


@task(log_prints=True)
def fastp(
    r1: str | PathLike, r2: str | PathLike, output_dir: str | PathLike, *, thread: int = 8
) -> FastpOutput:
    """运行Fastp软件对原始数据进行质控

    Args:
        r1: pair-end 文件 1
        r2: pair-end 文件 2
        output_dir: fastp输出目录
        thread: fastp 运行的进程数

    Returns:
        FastpOutput:
            r1: 清洗后的r1文件
            r2: 清洗后的r2文件
            json_report: json格式的指控报告
    """
    r1_file, r2_file, output_path, sample_name = get_path(r1, r2, output_dir=output_dir)
    client = docker.from_env()

    outputs = {}
    commands = [
        'fastp',
        f'-i /data/raw/{r1_file.name}',
        f'-I /data/raw/{r1_file.name}',
        f'-o /data/output/{sample_name}_fastp_R1.fq.gz',
        f'-O /data/output/{sample_name}_fastp_R2.fq.gz',
        f'-j /data/output/{sample_name}_fastp.json',
        f'-w {thread}',
    ]

    outputs['r1'] = output_path / f'{sample_name}_fastp_R1.fq.gz'
    outputs['r2'] = output_path / f'{sample_name}_fastp_R2.fq.gz'
    outputs['json_report'] = output_path / f'{sample_name}_fastp.json'

    client.containers.run(
        FASTP_IMAGE,
        ' '.join(commands),
        volumes={
            r1_file.absolute().parent: {'bind': '/data/raw', 'mode': 'rw'},
            output_path.absolute(): {'bind': '/data/output', 'mode': 'rw'},
        },
        auto_remove=True,
    )

    return FastpOutput.model_validate(outputs)
