from pathlib import Path

from celery import Task
from loguru import logger
from pydantic import BaseModel

from app.bioinfo.base_task import BaseBioTask
from app.core.config import get_settings
from app.utils.docker import run_container
from app.utils.input_format import get_path
from worker import celery_app


class FastpOutput(BaseModel):
    command: str
    r1: Path
    r2: Path
    json_report: Path
    log: Path


@celery_app.task(bind=True, pydantic=True, base=BaseBioTask)
def fastp(
    self: Task, *, r1: str, r2: str, output_dir: str, params: str = ''
) -> FastpOutput:
    """运行Fastp软件对原始数据进行质控

    Args:
        self:
        r1: pair-end 文件 1
        r2: pair-end 文件 2
        output_dir: fastp输出目录
        params: 其他参数

    Returns:
        r1: 清洗后的r1文件路径
        r2: 清洗后的r2文件路径
        json_report: json格式的质控报告路径
        log: 运行日志
    """
    logger.info(self.request.id)
    r1_file, r2_file, output_path, sample_name = get_path(r1, r2, output_dir=output_dir)

    outputs = {
        'r1': output_path / f'{sample_name}_fastp_R1.fq.gz',
        'r2': output_path / f'{sample_name}_fastp_R2.fq.gz',
        'json_report': output_path / f'{sample_name}_fastp.json',
        'log': output_path / 'fastp.log',
    }

    if all(path.exists() for path in outputs.values()):
        return FastpOutput.model_validate(outputs)

    commands = [
        'fastp',
        f'-i /data/raw/{r1_file.name}',
        f'-I /data/raw/{r2_file.name}',
        f'-o /data/output/{sample_name}_fastp_R1.fq.gz',
        f'-O /data/output/{sample_name}_fastp_R2.fq.gz',
        f'-j /data/output/{sample_name}_fastp.json',
        f'{params}',
        '&> /data/output/fastp.log',
    ]

    image_cfg = get_settings().docker.fastp
    run_container(
        image_cfg,
        commands=['/bin/bash', '-c', ' '.join(commands)],
        volumes_binds=[
            f'{r1_file.absolute().parent}:/data/raw',
            f'{output_path.absolute()}:/data/output',
        ],
    )

    return FastpOutput.model_validate(outputs)
