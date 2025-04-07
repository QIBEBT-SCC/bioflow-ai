import shutil
import tempfile
from os import PathLike
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from app.utils.config import get_settings
from app.utils.docker import run_container_async
from app.utils.input_format import get_path


class SpadesOutput(BaseModel):
    contig: Path
    log: Path


async def spades(
    r1: str | PathLike,
    r2: str | PathLike,
    output_dir: str | PathLike,
    *,
    mode: Literal['isolate', 'sc', 'bio', 'sewage', 'corona', 'rna'] = 'sc',
    thread: int = 64,
) -> SpadesOutput:
    r1_file, r2_file, output_path, sample_name = get_path(r1, r2, output_dir=output_dir)
    image_cfg = get_settings().docker.spades

    outputs = {'contig': output_path / 'contigs.fasta', 'log': output_path / 'spades.log'}

    if all(path.exists() for path in outputs.values()):
        return SpadesOutput.model_validate(outputs)

    with tempfile.TemporaryDirectory(prefix='prefect_spades_') as temp_dir:
        commands = [
            'spades.py',
            f'-1 /data/raw/{r1_file.name}',
            f'-2 /data/raw/{r2_file.name}',
            f'--{mode}',
            f'-t {thread}',
            '-o /data/output',
        ]

        await run_container_async(
            image_cfg.get_image(),
            commands=['/bin/bash', '-c', ' '.join(commands)],
            volumes_binds=[
                f'{r1_file.absolute().parent}:/data/raw',
                f'{Path(temp_dir).absolute()}:/data/output',
            ],
        )

        shutil.copy(Path(temp_dir) / 'contigs.fasta', output_path / 'contigs.fasta')
        shutil.copy(Path(temp_dir) / 'spades.log', output_path / 'spades.log')

    return SpadesOutput.model_validate(outputs)
