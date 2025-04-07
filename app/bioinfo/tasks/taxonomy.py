from pathlib import Path

from pydantic import BaseModel

from app.utils.config import get_settings
from app.utils.docker import run_container_async
from app.utils.input_format import get_path


class MetaPhlAnOutput(BaseModel):
    bowtie2out: Path
    taxon: Path
    log: Path


async def metaphlan(
    r1: str | Path, r2: str | Path, output_dir: str | Path
) -> MetaPhlAnOutput:
    r1_file, r2_file, output_path, sample_name = get_path(r1, r2, output_dir=output_dir)

    image_cfg = get_settings().docker.metaphlan
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
    ]

    await run_container_async(
        image_cfg.get_image(),
        commands=['/bin/bash', '-c', ' '.join(commands)],
        volumes_binds=[
            f'{r1_file.absolute().parent}:/data/raw',
            f'{output_path.absolute()}:/data/output',
            f'{db_path.absolute()}:/data/db',
        ],
    )

    return MetaPhlAnOutput.model_validate(outputs)
