from pathlib import Path
from typing import Optional, Any

from celery import Task
from loguru import logger
from pydantic import BaseModel

from app.bioinfo.base_task import BaseBioTask
from app.core.config import get_settings
from app.utils.docker import run_container
from app.utils.input_format import get_path
from worker import celery_app


class MetaPhlAnOutput(BaseModel):
    bowtie2out: Path
    taxon: Path
    log: Path


@celery_app.task(bind=True, pydantic=True, base=BaseBioTask)
def metaphlan(
    self: Task,
    result_dict: Optional[dict[str, Any]] = None,
    *,
    r1: Optional[str] = None,
    r2: Optional[str] = None,
    output_dir: str,
) -> MetaPhlAnOutput:
    if result_dict:
        r1 = result_dict.get('r1')
        r2 = result_dict.get('r2')

    logger.info(self.request.id)
    r1_file, r2_file, output_path, sample_name = get_path(r1, r2, output_dir=output_dir)

    tool_cfg = get_settings().tool
    db_path = Path(tool_cfg.METAPHLAN_DB)

    outputs = {
        'bowtie2out': output_path / f'{sample_name}.bowtie2.bz2',
        'taxon': output_path / f'{sample_name}_profiled_metagenome.txt',
        'log': output_path / 'metaphlan.log',
    }

    if all(path.exists() for path in outputs.values()):
        return MetaPhlAnOutput.model_validate(outputs)

    commands = [
        'metaphlan',
        f'/data/raw/{r1_file.name},/data/raw/{r2_file.name}',
        '--input_type fastq',
        f'--bowtie2out /data/output/{sample_name}.bowtie2.bz2',
        f'-o /data/output/{sample_name}_profiled_metagenome.txt',
        '--bowtie2db /data/db',
        f'--nproc {tool_cfg.THREADS}',
        '&> /data/output/metaphlan.log',
    ]

    image_cfg = get_settings().docker.metaphlan
    run_container(
        image_cfg,
        commands=['/bin/bash', '-c', ' '.join(commands)],
        volumes_binds=[
            f'{r1_file.absolute().parent}:/data/raw',
            f'{output_path.absolute()}:/data/output',
            f'{db_path.absolute()}:/data/db',
        ],
    )

    return MetaPhlAnOutput.model_validate(outputs)


class TaxonInfo(BaseModel):
    ncbi_name: str
    ncbi_id: int
    level: str
