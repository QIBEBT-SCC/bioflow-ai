import shutil
import tempfile
from os import PathLike
from pathlib import Path
from typing import Literal

import docker
from prefect import task
from pydantic import BaseModel

from app.utils.config import get_settings
from app.utils.input_format import get_path


@task(log_prints=True)
def rename(input_file: str | PathLike, output_dir: str | PathLike) -> Path:
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

    client = docker.from_env()
    commands = [
        'bash -c',
        '"gzip',
        f'-cd /data/raw/{input_path.name}',
        '|',
        'fastx_renamer',
        '-n COUNT',
        '-i /dev/stdin',
        '-Q33',
        '-z',
        f'-o /data/output/{new_name}"',
    ]

    image_cfg = get_settings().docker.fastx
    client.images.pull(image_cfg.REPOSITORY, image_cfg.TAG)
    client.containers.run(
        image_cfg.get_image(),
        ' '.join(commands),
        volumes={
            input_path.absolute().parent: {'bind': '/data/raw', 'mode': 'rw'},
            output_path.absolute(): {'bind': '/data/output', 'mode': 'rw'},
        },
        auto_remove=True,
    )

    return output_path / new_name


class BBNormOutput(BaseModel):
    r1: Path
    r2: Path


@task
def bb_norm(
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

    client = docker.from_env()
    commands = [
        'bbnorm.sh',
        f'in=/data/{r1_file.name}',
        f'in2=/data/{r2_file.name}',
        f'out=/data/{sample_name}_R1_norm.fq.gz',
        f'out2=/data/{sample_name}_R2_norm.fq.gz',
        f'threads={thread}',
        f'-Xmx{memory}g',
    ]

    image_cfg = get_settings().docker.bbmap
    client.images.pull(image_cfg.REPOSITORY, image_cfg.TAG)
    client.containers.run(
        image_cfg.get_image(),
        ' '.join(commands),
        volumes={
            r1_file.absolute().parent: {'bind': '/data', 'mode': 'rw'},
            output_path.absolute(): {'bind': '/data/output', 'mode': 'rw'},
        },
        auto_remove=True,
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
def spades(
    r1: str | PathLike,
    r2: str | PathLike,
    output_dir: str | PathLike,
    *,
    mode: Literal['isolate', 'sc', 'bio', 'sewage', 'corona', 'rna'] = 'sc',
    thread: int = 64,
):
    r1_file, r2_file, output_path, sample_name = get_path(r1, r2, output_dir=output_dir)

    client = docker.from_env()
    with tempfile.TemporaryDirectory() as temp_dir:
        command = [
            'spades.py',
            '-1',
            f'/data/{r1_file.name}',
            '-2',
            f'/data/{r2_file.name}',
            f'--{mode}',
            '-t',
            f'{thread}',
            '--gfa11',
            '-o',
            '/data/output',
        ]

        image_cfg = get_settings().docker.spades
        client.images.pull(image_cfg.REPOSITORY, image_cfg.TAG)
        client.containers.run(
            image_cfg.get_image(),
            command,
            volumes={
                r1_file.absolute().parent: {'bind': '/data', 'mode': 'rw'},
                output_path.absolute(): {'bind': '/data/output', 'mode': 'rw'},
            },
            auto_remove=True,
        )

        shutil.copy(Path(temp_dir) / 'contigs.fasta', output_path / 'contigs.fasta')
        shutil.copy(Path(temp_dir) / 'scaffolds.fasta', output_path / 'scaffolds.fasta')
        shutil.copy(Path(temp_dir) / 'spades.log', output_path / 'spades.log')

    return SpadesOutput(
        contig=output_path / 'contigs.fasta',
        scaffolds=output_path / 'scaffolds.fasta',
        log=output_path / 'spades.log',
    )
