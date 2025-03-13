import os
import shutil
import tempfile
from os import PathLike
from pathlib import Path
from typing import Optional

import docker
from prefect import task

from app.bioinfo.tasks import FASTP_IMAGE


@task(log_prints=True)
def fastp(
    r1: str | PathLike, r2: str | PathLike, output_dir: str | PathLike
) -> tuple[PathLike, Optional[PathLike]]:
    r1_file = Path(r1) if isinstance(r1, str) else r1
    r2_file = Path(r2) if isinstance(r2, str) else r2
    output_path = Path(output_dir) if isinstance(output_dir, str) else output_dir

    sample_name = os.path.commonprefix([r1_file.name, r2_file.name]).strip('_')
    client = docker.from_env()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_r1 = Path(temp_dir) / f'{sample_name}_R1.fq.gz'
        shutil.copy(r1_file, temp_r1)
        command = [
            'fastp',
            '-i',
            f'/data/{sample_name}_R1.fq.gz',
            '-o',
            f'/data/output/{sample_name}_fastp_R1.fq.gz',
        ]
        if r2_file:
            temp_r2 = Path(temp_dir) / f'{sample_name}_R2.fq.gz'
            shutil.copy(r2_file, temp_r2)
            command.extend(
                [
                    '-I',
                    f'/data/{sample_name}_R2.fq.gz',
                    '-O',
                    f'/data/output/{sample_name}_fastp_R2.fq.gz',
                ]
            )
        command.extend(
            [
                '-h',
                f'/data/output/{sample_name}_fastp.html',
                '-j',
                f'/data/output/{sample_name}_fastp.json',
                '-w',
                '8',
            ]
        )

        client.containers.run(
            FASTP_IMAGE,
            command,
            volumes={
                temp_dir: {'bind': '/data', 'mode': 'rw'},
                output_path.absolute(): {'bind': '/data/output', 'mode': 'rw'},
            },
            auto_remove=True,
        )

        return (
            output_path / f'{sample_name}_fastp_R1.fq.gz',
            output_path / f'{sample_name}_fastp_R2.fq.gz',
        )
