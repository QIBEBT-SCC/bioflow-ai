from pathlib import Path
from typing import Optional

from celery import Task
from pydantic import BaseModel

from app.bioinfo.base_task import BaseBioTask
from app.utils.config import get_settings
from app.utils.docker import run_container_async, run_container
from app.utils.input_format import get_path
from worker import celery_app


class DiamondBlastOutput(BaseModel):
    daa_file: Path
    log: Path


@celery_app.task(bind=True, pydantic=True, base=BaseBioTask)
def diamond_blast(
    self: Task, *, input_path: str | Path, output_dir: str | Path, params: Optional[str] = None
) -> DiamondBlastOutput:
    """Diamond BLAST分析

    Celery版本返回字符串路径而不是Path对象，以便序列化

    Args:
        self:
        input_path: 输入文件路径
        output_dir: 输出目录
        params: 其他参数

    Returns:
        str: 输出文件的路径字符串
    """
    input_file = input_path if isinstance(input_path, Path) else Path(input_path)
    output_path = output_dir if isinstance(output_dir, Path) else Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    outputs = {
        'daa_file': output_path / f'{input_file.name.removesuffix(".fq.gz")}.daa',
        'log': output_path / 'blastx.log',
    }

    if all(path.exists() for path in outputs.values()):
        return DiamondBlastOutput.model_validate(outputs)

    db_path = Path(get_settings().tool.DIAMOND_DB)

    commands = [
        'diamond blastx',
        f'-q /data/raw/{input_file.name}',
        f'-o /data/output/{input_file.name.removesuffix(".fq.gz")}.daa',
        '--fast',
        '-f 100',
        '-d /data/db/nr',
        '--evalue 1e-5',
        '--top 1',
        f'-p {get_settings().tool.THREADS}',
        '&> /data/output/blastx.log',
    ]

    image_cfg = get_settings().docker.diamond
    run_container(
        image_cfg,
        commands=['/bin/bash', '-c', ' '.join(commands)],
        volumes_binds=[
            f'{input_file.absolute().parent}:/data/raw',
            f'{output_path.absolute()}:/data/output',
            f'{db_path.absolute()}:/data/db',
        ],
    )

    return DiamondBlastOutput.model_validate(outputs)


class MegaPrepareOutput(BaseModel):
    megan_rma: Path
    log: Path


@celery_app.task(bind=True, pydantic=True, base=BaseBioTask)
async def megan_prepare(
    self: Task, *, daa_r1: str | Path, daa_r2: str | Path, output_dir: str | Path
) -> MegaPrepareOutput:
    """Computes a MEGAN .rma6 file from a DIAMOND .daa file

    Args:
        self:
        daa_r1: Input r1 DAA file
        daa_r2: Input r2 DAA file
        output_dir:

    Returns:

    """

    r1_file, r2_file, output_path, sample_name = get_path(daa_r1, daa_r2, output_dir=output_dir)

    outputs = {
        'megan_rma': output_path / f'{sample_name}.rma6',
        'log': output_path / 'megan_prepare.log',
    }
    if all(path.exists() for path in outputs.values()):
        return MegaPrepareOutput.model_validate(outputs)

    tool_cfg = get_settings().tool
    db_path = Path(tool_cfg.MEGAN_MAPPING_DB)

    commands = ' '.join(
        [
            f'echo -Xmx{tool_cfg.MAX_MEM} > /usr/local/opt/megan-6.25.9/MEGAN.vmoptions &&',
            'daa2rma',
            f'--in /data/raw/{r1_file.name} /data/raw/{r2_file.name}',
            f'--out /data/output/{sample_name}.rma6',
            f'--mapDB /data/db/{db_path.name}',
            '--paired',
            f'-t {tool_cfg.THREADS}',
            '&> /data/output/megan_prepare.log',
        ]
    )

    image_cfg = get_settings().docker.megan
    await run_container_async(
        image_cfg.get_image(),
        commands=['bash', '-c', commands],
        volumes_binds=[
            f'{r1_file.absolute().parent}:/data/raw',
            f'{output_path.absolute()}:/data/output',
            f'{db_path.absolute().parent}:/data/db',
        ],
    )

    return MegaPrepareOutput.model_validate(outputs)
