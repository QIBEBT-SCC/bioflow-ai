import shutil
import tempfile
from os import PathLike
from pathlib import Path
from typing import Literal

from prefect import task
from pydantic import BaseModel

from app.utils.config import get_settings
from app.utils.docker import run_container_async
from app.utils.input_format import get_path


@task(log_prints=True)
async def rename(input_file: str | PathLike, output_dir: str | PathLike) -> Path:
    """对fastq文件进行重命名

    Args:
        input_file:
        output_dir:
    """
    input_path = Path(input_file) if isinstance(input_file, str) else input_file
    output_path = Path(output_dir) if isinstance(output_dir, str) else output_dir

    output_path.mkdir(parents=True, exist_ok=True)

    origin_name = input_path.name.split('.')[0]
    new_name = input_path.name.replace(origin_name, f'{origin_name}_rename')

    commands = [
        'gzip',
        f'-cd /data/raw/{input_path.name}',
        '|',
        'fastx_renamer',
        '-n COUNT',
        '-i /dev/stdin',
        '-Q33',
        '-z',
        '-o',
        f'/data/output/{new_name}',
    ]

    image_cfg = get_settings().docker.fastx
    await run_container_async(
        image_cfg.get_image(),
        commands=['/bin/bash', '-c', ' '.join(commands)],
        volumes_binds=[
            f'{input_path.absolute().parent}:/data/raw',
            f'{output_path.absolute()}:/data/output',
        ],
    )

    return output_path / new_name


class BBNormOutput(BaseModel):
    r1: Path
    r2: Path


@task
async def bb_norm(
    r1: str | PathLike,
    r2: str | PathLike,
    output_dir: str | PathLike,
    *,
    thread: int = 8,
    memory: int = 16,
) -> BBNormOutput:
    """

    Args:
        r1:
        r2:
        output_dir:
        thread:
        memory:

    Returns:

    """
    r1_file, r2_file, output_path, sample_name = get_path(r1, r2, output_dir=output_dir)
    image_cfg = get_settings().docker.bbmap

    commands = [
        'bbnorm.sh',
        f'in=/data/raw/{r1_file.name}',
        f'in2=/data/raw/{r2_file.name}',
        f'out=/data/output/{sample_name}_R1_norm.fq.gz',
        f'out2=/data/output/{sample_name}_R2_norm.fq.gz',
        f'threads={thread}',
        f'-Xmx{memory}g',
    ]

    await run_container_async(
        image_cfg.get_image(),
        commands=commands,
        volumes_binds=[
            f'{r1_file.absolute().parent}:/data/raw',
            f'{output_path.absolute()}:/data/output',
        ],
    )

    return BBNormOutput(
        r1=output_path / f'{sample_name}_R1_norm.fq.gz',
        r2=output_path / f'{sample_name}_R2_norm.fq.gz',
    )


class SpadesOutput(BaseModel):
    contig: Path
    scaffolds: Path
    log: Path


@task
async def spades(
    r1: str | PathLike,
    r2: str | PathLike,
    output_dir: str | PathLike,
    *,
    mode: Literal['isolate', 'sc', 'bio', 'sewage', 'corona', 'rna'] = 'sc',
    thread: int = 64,
):
    r1_file, r2_file, output_path, sample_name = get_path(r1, r2, output_dir=output_dir)
    image_cfg = get_settings().docker.spades

    with tempfile.TemporaryDirectory() as temp_dir:
        commands = [
            'spades.py',
            f'-1 /data/raw/{r1_file.name}',
            f'-2 /data/raw/{r2_file.name}',
            f'--{mode}',
            f'-t {thread}',
            '--gfa11',
            '-o /data/output',
        ]

        await run_container_async(
            image_cfg.get_image(),
            commands=[' '.join(commands)],
            volumes_binds=[
                f'{r1_file.absolute().parent}:/data/raw',
                f'{output_path.absolute()}:/data/output',
            ],
        )

        shutil.copy(Path(temp_dir) / 'contigs.fasta', output_path / 'contigs.fasta')
        shutil.copy(Path(temp_dir) / 'scaffolds.fasta', output_path / 'scaffolds.fasta')
        shutil.copy(Path(temp_dir) / 'spades.log', output_path / 'spades.log')

    return SpadesOutput(
        contig=output_path / 'contigs.fasta',
        scaffolds=output_path / 'scaffolds.fasta',
        log=output_path / 'spades.log',
    )
