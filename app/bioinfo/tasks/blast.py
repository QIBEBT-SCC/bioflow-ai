from pathlib import Path

from pydantic import BaseModel

from app.utils.config import get_settings
from app.utils.docker import run_container_async
from app.utils.input_format import get_path


async def diamond_blast(input_path: str | Path, output_dir: str | Path) -> str:
    """Diamond BLAST分析

    Celery版本返回字符串路径而不是Path对象，以便序列化

    Args:
        input_path: 输入文件路径
        output_dir: 输出目录

    Returns:
        str: 输出文件的路径字符串
    """
    input_file = input_path if isinstance(input_path, Path) else Path(input_path)
    output_path = output_dir if isinstance(output_dir, Path) else Path(output_dir)
    output_file = output_path / f'{input_file.name.removesuffix(".fq.gz")}.daa'
    db_path = Path(get_settings().tool.DIAMOND_DB)

    if output_file.exists():
        return str(output_file)

    output_path.mkdir(parents=True, exist_ok=True)
    image_cfg = get_settings().docker.diamond

    commands = [
        'diamond blastx',
        f'-q /data/raw/{input_file.name}',
        f'-o /data/output/{output_file.name}',
        '--fast',
        '-f 100',
        '-d /data/db/nr',
        '--evalue 1e-5',
        '--top 1',
        f'-p {get_settings().tool.THREADS}',
    ]

    await run_container_async(
        image_cfg.get_image(),
        commands=['/bin/bash', '-c', ' '.join(commands)],
        volumes_binds=[
            f'{input_file.absolute().parent}:/data/raw',
            f'{output_path.absolute()}:/data/output',
            f'{db_path.absolute()}:/data/db',
        ],
    )

    return str(output_file)


class MegaPrepareOutput(BaseModel):
    megan_rma: Path
    log: Path


async def megan_prepare(
    daa_r1: str | Path,
    daa_r2: str | Path,
    output_dir: str | Path,
    *,
    ms: float = 50.0,
    me: float = 0.01,
    mpi: float = 0.0,
    top: float = 10.0,
    supp: float = 0.01,
    alg: str = 'naive',
) -> MegaPrepareOutput:
    """Computes a MEGAN .rma6 file from a DIAMOND .daa file

    Args:
        daa_r1: Input r1 DAA file
        daa_r2: Input r2 DAA file
        output_dir:
        ms: Min score. Default value: 50.0.
        me: Max expected. Default value: 0.01.
        mpi: Min percent identity. Default value: 0.0.
        top: Top percent. Default value: 10.0.
        supp: Min support as percent of assigned reads (0==off). Default value: 0.01.
        alg: Set the LCA algorithm to use for taxonomic assignment. Default value: naive. Legal values: naive, weighted, longReads

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
    image_cfg = get_settings().docker.megan

    commands = ' '.join(
        [
            f'echo -Xmx{tool_cfg.MAX_MEM} > /usr/local/opt/megan-6.25.9/MEGAN.vmoptions &&',
            'daa2rma',
            f'--in /data/raw/{r1_file.name} /data/raw/{r2_file.name}',
            f'--out /data/output/{sample_name}.rma6',
            f'--mapDB /data/db/{db_path.name}',
            '--paired',
            f'--minScore {ms}',
            f'--maxExpected {me}',
            f'--minPercentIdentity {mpi}',
            f'--topPercent {top}',
            f'--minSupportPercent {supp}',
            f'--lcaAlgorithm {alg}',
            f'-t {tool_cfg.THREADS}',
            '&> /data/output/megan_prepare.log',
        ]
    )

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
