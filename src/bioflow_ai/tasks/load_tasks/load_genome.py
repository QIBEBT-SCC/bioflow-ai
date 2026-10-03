from pathlib import Path
from typing import Any

from celery import Task
from loguru import logger
from sqlalchemy.orm import Session

import bioflow_ai.crud.resource.reference_genome as genome_crud
from bioflow_ai.cli.worker import celery_app
from bioflow_ai.core.config import get_settings
from bioflow_ai.core.session import engine
from bioflow_ai.models.resource import ReferenceGenomes
from bioflow_ai.schemas.react_flow import TaskInput
from bioflow_ai.schemas.resource import GenomeUpdate
from bioflow_ai.schemas.tool_config import ImageConfig
from bioflow_ai.schemas.workflow_config import ToolOutput
from bioflow_ai.services.resource import reference_genome as genome_service
from bioflow_ai.services.workflow.execution_context import get_node_output
from bioflow_ai.utils.api.ncbi_datasets import (
    NCBIAssemblyDataReport,
    download_genome_data,
    get_species_info,
    search_reference_genomes,
    search_reference_genomes_by_tax_id,
)
from bioflow_ai.utils.container import ContainerMount, run_command
from bioflow_ai.utils.network import download_gz_files

GENOME_PATH = get_settings().persistent.storage.GENOME_PATH
MAX_THREAD = get_settings().persistent.resource.MAX_THREAD

HG38_FILES: dict[str, dict[str, str]] = {
    'genome_fasta': {
        'url': 'https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/GRCh38.primary_assembly.genome.fa.gz',
        'filename': 'GRCh38.primary_assembly.genome.fa',
    },
    'annotation_gff': {
        'url': 'https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/gencode.v49.primary_assembly.annotation.gff3.gz',
        'filename': 'gencode.v49.primary_assembly.annotation.gff3',
    },
    'annotation_gtf': {
        'url': 'https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/gencode.v49.primary_assembly.annotation.gtf.gz',
        'filename': 'gencode.v49.primary_assembly.annotation.gtf',
    },
}

MM39_FILES: dict[str, dict[str, str]] = {
    'genome_fasta': {
        'url': 'https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M38/GRCm39.primary_assembly.genome.fa.gz',
        'filename': 'GRCm39.primary_assembly.genome.fa',
    },
    'annotation_gff': {
        'url': 'https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M38/gencode.vM38.primary_assembly.annotation.gff3.gz',
        'filename': 'gencode.vM38.primary_assembly.annotation.gff3',
    },
    'annotation_gtf': {
        'url': 'https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M38/gencode.vM38.primary_assembly.annotation.gtf.gz',
        'filename': 'gencode.vM38.primary_assembly.annotation.gtf',
    },
}


def build_bowtie2_index(genome_fasta: str) -> str:
    """构建 bowtie2 索引."""
    genome_path = Path(genome_fasta)
    index_path = genome_path.parent / 'bowtie2'
    index_path.mkdir(exist_ok=True, parents=True)

    mounts = [
        ContainerMount(read_only=False, source=str(genome_path.parent), target='/data/raw'),
        ContainerMount(read_only=False, source=str(index_path), target='/data/output'),
    ]

    image_cfg = ImageConfig.from_str('quay.io/biocontainers/bowtie2:2.5.4--he96a11b_6')
    status, output = run_command(
        image_cfg,
        mounts,
        f'bowtie2-build --threads {MAX_THREAD} /data/raw/{genome_path.name} /data/output/{genome_path.stem}',
    )

    if status == 0:
        logger.success('build bowtie2 index succeed')
    else:
        logger.error(output)
        raise ValueError(output)

    return f'{index_path}/{genome_path.stem}'


def build_bwa_index(genome_fasta: str, *, is_long: bool = False) -> str:
    """构建 bwa 索引."""
    genome_path = Path(genome_fasta)

    mounts = [
        ContainerMount(read_only=False, source=str(genome_path.parent), target='/data/raw'),
    ]

    image_cfg = ImageConfig.from_str('quay.io/biocontainers/bwa:0.7.19--h577a1d6_1')
    command = (
        f'bwa index -a bwtsw /data/raw/{genome_path.name}' if is_long else f'bwa index /data/raw/{genome_path.name}'
    )
    status, output = run_command(
        image_cfg,
        mounts,
        command,
    )

    if status == 0:
        logger.success('build bwa index succeed')
    else:
        logger.error(output)
        raise ValueError(output)

    return f'{genome_path}'


def build_hisat2_index(genome_fasta: str) -> str:
    """构建 hisat2 索引."""
    genome_path = Path(genome_fasta)
    index_path = genome_path.parent / 'hisat2'
    index_path.mkdir(exist_ok=True, parents=True)

    mounts = [
        ContainerMount(read_only=False, source=str(genome_path.parent), target='/data/raw'),
        ContainerMount(read_only=False, source=str(index_path), target='/data/output'),
    ]

    image_cfg = ImageConfig.from_str('quay.io/biocontainers/hisat2:2.2.1--h503566f_8')
    status, output = run_command(
        image_cfg,
        mounts,
        f'hisat2-build -p {MAX_THREAD} /data/raw/{genome_path.name} /data/output/{genome_path.stem}',
    )

    if status == 0:
        logger.success('build hisat2 index succeed')
    else:
        logger.error(output)
        raise ValueError(output)

    return f'{index_path}/{genome_path.stem}'


def build_star_index(genome_fasta: str, gff: str) -> str:
    """构建 STAR 索引."""
    genome_path = Path(genome_fasta)
    gff_path = Path(gff)
    index_path = genome_path.parent / 'star'
    index_path.mkdir(exist_ok=True, parents=True)

    mounts = [
        ContainerMount(read_only=False, source=str(genome_path.parent), target='/data/raw'),
        ContainerMount(read_only=False, source=str(index_path), target='/data/output'),
    ]

    image_cfg = ImageConfig.from_str('quay.io/biocontainers/star:2.7.11b--h5ca1c30_7')
    status, output = run_command(
        image_cfg,
        mounts,
        f'STAR --runThreadN {MAX_THREAD} --runMode genomeGenerate --genomeDir /data/output --genomeFastaFiles /data/raw/{genome_path.name} --sjdbGTFfile /data/raw/{gff_path.name}',
    )

    if status == 0:
        logger.success('build STAR index succeed')
    else:
        logger.error(output)
        raise ValueError(output)

    return f'{index_path}'


def build_minimap2_index(genome_fasta: str) -> str:
    """构建 minimap2 索引."""
    genome_path = Path(genome_fasta)
    index_path = genome_path.parent / 'minimap2'
    index_path.mkdir(exist_ok=True, parents=True)

    mounts = [
        ContainerMount(read_only=False, source=str(genome_path.parent), target='/data/raw'),
        ContainerMount(read_only=False, source=str(index_path), target='/data/output'),
    ]

    image_cfg = ImageConfig.from_str('quay.io/biocontainers/minimap2:2.30--h577a1d6_0')
    status, output = run_command(
        image_cfg,
        mounts,
        f'minimap2 -t {MAX_THREAD} -d /data/output/{genome_path.stem}.mmi /data/raw/{genome_path.name} ',
    )

    if status == 0:
        logger.success('build minimap2 index succeed')
    else:
        logger.error(output)
        raise ValueError(output)

    return f'{index_path}/{genome_path.stem}.mmi'


def _create_genome_record(
    session: Session, genome: NCBIAssemblyDataReport, species_info: Any, alias_name: str | None = None
) -> ReferenceGenomes:
    """创建基因组数据库记录."""
    download_result = download_genome_data(genome.accession, GENOME_PATH, include_annotation=True, timeout=3600)

    genome_record = ReferenceGenomes(
        ncbi_tax_id=species_info.tax_id,
        name=species_info.organism_name,
        accession=genome.accession,
        genome_sequences=download_result['genome_fasta'],
        annotation_gff=download_result['annotation_gff'],
        annotation_gtf=download_result['annotation_gtf'],
    )

    if alias_name and alias_name != genome_record.name:
        genome_record.aliases = alias_name

    return genome_service.create_genome(session, genome_record)


def load_ncbi_genome(
    *,
    inputs: dict[str, TaskInput],
    required_index: list[str],
) -> ToolOutput:
    """下载基因组任务实例."""
    load_result = {}
    with Session(engine) as session:
        if 'ncbi_tax_id' in inputs:
            tax_id, _ = get_node_output(inputs['ncbi_tax_id'].source_uid, inputs['ncbi_tax_id'].source_k)
            # 首先尝试通过tax_id在数据库中查找
            genome_record = genome_crud.get_by_taxid(session, int(tax_id))

            if not genome_record:
                # 如果数据库中没有, 通过API搜索并下载
                genome = search_reference_genomes_by_tax_id(int(tax_id))
                species_info = get_species_info(genome.organism.tax_id)
                genome_record = _create_genome_record(session, genome, species_info)
            else:
                logger.debug(f'Found existed genome record for tax_id: "{tax_id}"')
        elif 'species_name' in inputs:
            species_name, _ = get_node_output(inputs['species_name'].source_uid, inputs['species_name'].source_k)
            genome_records = genome_crud.search(session, species_name)

            if not genome_records:
                genome = search_reference_genomes(species_name)
                species_info = get_species_info(genome.organism.tax_id)

                if genome_record := genome_crud.get_by_taxid(session, species_info.tax_id):
                    # pyrefly: ignore [bad-argument-type]
                    genome_update = GenomeUpdate(id=genome_record.id, aliases=f'{genome_record.aliases}|{species_name}')
                    genome_record = genome_service.update_genome(session, genome_update)
                else:
                    genome_record = _create_genome_record(session, genome, species_info, species_name)
            else:
                logger.debug(f'Found existed genome record for species: "{species_name}"')
                genome_record = genome_records[0]

        else:
            error_msg = "Either 'ncbi_tax_id' or 'species_name' must be provided in inputs"
            raise ValueError(error_msg)

        load_result['genome_fasta'] = genome_record.genome_sequences
        load_result['annotation_gff'] = genome_record.annotation_gff
        load_result['annotation_gtf'] = genome_record.annotation_gtf

        # 定义索引构建器映射
        index_builders = {
            'bowtie2_index': lambda: build_bowtie2_index(genome_record.genome_sequences),
            'bwa_index': lambda: build_bwa_index(genome_record.genome_sequences),
            'hisat2_index': lambda: build_hisat2_index(genome_record.genome_sequences),
            'star_index': lambda: build_star_index(genome_record.genome_sequences, genome_record.annotation_gff),
            'minimap2_index': lambda: build_minimap2_index(genome_record.genome_sequences),
        }

        # 处理每个需要的索引
        logger.info(f'同时加载索引: {required_index}')
        for index_name in required_index:
            if index_name in index_builders:
                # 检查数据库中是否已存在该索引
                existing_index = getattr(genome_record, index_name, None)
                if existing_index:
                    load_result[index_name] = existing_index
                else:
                    # 构建新索引
                    new_index = index_builders[index_name]()
                    # 更新数据库记录
                    update_data = {index_name: new_index}
                    # pyrefly: ignore [bad-argument-type]
                    genome_update = GenomeUpdate(id=genome_record.id, **update_data)
                    genome_record = genome_service.update_genome(session, genome_update)
                    load_result[index_name] = getattr(genome_record, index_name)

    return ToolOutput(result=load_result)


def load_hg38(
    *,
    required_index: list[str],
) -> ToolOutput:
    """下载 HG38 基因组任务实例(从 Gencode 下载, 不依赖数据库)."""
    hg38_dir = Path(GENOME_PATH) / 'HG38'
    files = download_gz_files(HG38_FILES, hg38_dir)
    load_result = {
        'genome_fasta': files['genome_fasta'],
        'annotation_gff': files['annotation_gff'],
        'annotation_gtf': files['annotation_gtf'],
    }

    genome_fasta = files['genome_fasta']
    annotation_gff = files['annotation_gff']
    hg38_dir = Path(genome_fasta).parent

    index_builders = {
        'bowtie2_index': lambda: build_bowtie2_index(genome_fasta),
        'bwa_index': lambda: build_bwa_index(genome_fasta, is_long=True),
        'hisat2_index': lambda: build_hisat2_index(genome_fasta),
        'star_index': lambda: build_star_index(genome_fasta, annotation_gff),
        'minimap2_index': lambda: build_minimap2_index(genome_fasta),
    }

    logger.debug(f'加载索引: {required_index}')
    for index_name in required_index:
        if index_name in index_builders:
            if _index_exists(hg38_dir, index_name, Path(genome_fasta)):
                # 根据索引类型推导已有路径
                load_result[index_name] = _get_existing_index_path(hg38_dir, index_name, Path(genome_fasta))
                logger.debug(f'{index_name} already exists, skipping build')
            else:
                load_result[index_name] = index_builders[index_name]()

    return ToolOutput(result=load_result)


def load_mm39(
    *,
    required_index: list[str],
) -> ToolOutput:
    """下载 MM39 基因组任务实例(从 Gencode 下载, 不依赖数据库)."""
    mm39_dir = Path(GENOME_PATH) / 'MM39'
    files = download_gz_files(MM39_FILES, mm39_dir)
    load_result = {
        'genome_fasta': files['genome_fasta'],
        'annotation_gff': files['annotation_gff'],
        'annotation_gtf': files['annotation_gtf'],
    }

    genome_fasta = files['genome_fasta']
    annotation_gff = files['annotation_gff']
    genome_dir = Path(genome_fasta).parent

    index_builders = {
        'bowtie2_index': lambda: build_bowtie2_index(genome_fasta),
        'bwa_index': lambda: build_bwa_index(genome_fasta, is_long=True),
        'hisat2_index': lambda: build_hisat2_index(genome_fasta),
        'star_index': lambda: build_star_index(genome_fasta, annotation_gff),
        'minimap2_index': lambda: build_minimap2_index(genome_fasta),
    }

    logger.debug(f'加载索引: {required_index}')
    for index_name in set(required_index):
        if index_name in index_builders:
            if _index_exists(genome_dir, index_name, Path(genome_fasta)):
                load_result[index_name] = _get_existing_index_path(genome_dir, index_name, Path(genome_fasta))
                logger.debug(f'{index_name} already exists, skipping build')
            else:
                load_result[index_name] = index_builders[index_name]()

    return ToolOutput(result=load_result)


def _index_exists(genome_dir: Path, index_name: str, genome_fasta: Path) -> bool:
    """通过文件系统检查索引是否已构建."""
    checks = {
        'bowtie2_index': lambda: (
            any((genome_dir / 'bowtie2').iterdir()) if (genome_dir / 'bowtie2').exists() else False
        ),
        'bwa_index': lambda: genome_fasta.with_suffix('.fa.bwt').exists(),
        'hisat2_index': lambda: any((genome_dir / 'hisat2').iterdir()) if (genome_dir / 'hisat2').exists() else False,
        'star_index': lambda: any((genome_dir / 'star').iterdir()) if (genome_dir / 'star').exists() else False,
        'minimap2_index': lambda: (
            any((genome_dir / 'minimap2').glob('*.mmi')) if (genome_dir / 'minimap2').exists() else False
        ),
    }
    checker = checks.get(index_name)
    return checker() if checker else False


def _get_existing_index_path(genome_dir: Path, index_name: str, genome_fasta: Path) -> str:
    """返回已存在索引的路径."""
    paths = {
        'bowtie2_index': lambda: str(genome_dir / 'bowtie2' / genome_fasta.stem),
        'bwa_index': lambda: str(genome_fasta),
        'hisat2_index': lambda: str(genome_dir / 'hisat2' / genome_fasta.stem),
        'star_index': lambda: str(genome_dir / 'star'),
        'minimap2_index': lambda: str(genome_dir / 'minimap2' / f'{genome_fasta.stem}.mmi'),
    }
    return paths[index_name]()


@celery_app.task(bind=True, pydantic=True)
def download_genome_admin_task(
    self: Task,  # noqa: ARG001
    *,
    ncbi_tax_id: int | None = None,
    species_name: str | None = None,
    required_index: list[str],
) -> dict[str, Any]:
    """从 NCBI 下载参考基因组管理员直接调用版, 不依赖 Redis TaskInput."""
    load_result: dict[str, Any] = {}
    with Session(engine) as session:
        if ncbi_tax_id is not None:
            genome_record = genome_crud.get_by_taxid(session, ncbi_tax_id)
            if not genome_record:
                genome = search_reference_genomes_by_tax_id(ncbi_tax_id)
                species_info = get_species_info(genome.organism.tax_id)
                genome_record = _create_genome_record(session, genome, species_info)
            else:
                logger.info(f'Found existing genome record for tax_id: "{ncbi_tax_id}"')
        elif species_name is not None:
            genome_records = genome_crud.search(session, species_name)
            if not genome_records:
                genome = search_reference_genomes(species_name)
                species_info = get_species_info(genome.organism.tax_id)
                if genome_record := genome_crud.get_by_taxid(session, species_info.tax_id):
                    genome_update = GenomeUpdate(
                        # pyrefly: ignore [bad-argument-type]
                        id=genome_record.id,
                        aliases=f'{genome_record.aliases}|{species_name}',
                    )
                    genome_record = genome_service.update_genome(session, genome_update)
                else:
                    genome_record = _create_genome_record(session, genome, species_info, species_name)
            else:
                logger.info(f'Found existing genome record for species: "{species_name}"')
                genome_record = genome_records[0]
        else:
            error_msg = "Either 'ncbi_tax_id' or 'species_name' must be provided."
            raise ValueError(error_msg)

        load_result['genome_fasta'] = genome_record.genome_sequences
        load_result['annotation_gff'] = genome_record.annotation_gff
        load_result['annotation_gtf'] = genome_record.annotation_gtf
        load_result['genome_id'] = genome_record.id

        index_builders = {
            'bowtie2_index': lambda: build_bowtie2_index(genome_record.genome_sequences),
            'bwa_index': lambda: build_bwa_index(genome_record.genome_sequences),
            'hisat2_index': lambda: build_hisat2_index(genome_record.genome_sequences),
            'star_index': lambda: build_star_index(genome_record.genome_sequences, genome_record.annotation_gff),
            'minimap2_index': lambda: build_minimap2_index(genome_record.genome_sequences),
        }

        for index_name in required_index:
            if index_name in index_builders:
                existing_index = getattr(genome_record, index_name, None)
                if existing_index:
                    load_result[index_name] = existing_index
                else:
                    new_index = index_builders[index_name]()
                    update_data = {index_name: new_index}
                    # pyrefly: ignore [bad-argument-type]
                    genome_update = GenomeUpdate(id=genome_record.id, **update_data)
                    genome_record = genome_service.update_genome(session, genome_update)
                    load_result[index_name] = getattr(genome_record, index_name)

    return load_result


@celery_app.task(bind=True, pydantic=True)
def build_genome_index_admin_task(
    self: Task,  # noqa: ARG001
    *,
    genome_id: int,
    required_index: list[str],
) -> dict[str, Any]:
    """为已存在的参考基因组构建索引直接调用版."""
    load_result: dict[str, Any] = {}
    with Session(engine) as session:
        genome_record = genome_crud.get(session, genome_id)
        if not genome_record:
            error_msg = f'Genome with id={genome_id} not found.'
            raise ValueError(error_msg)

        load_result['genome_id'] = genome_record.id
        load_result['name'] = genome_record.name

        index_builders = {
            'bowtie2_index': lambda: build_bowtie2_index(genome_record.genome_sequences),
            'bwa_index': lambda: build_bwa_index(genome_record.genome_sequences),
            'hisat2_index': lambda: build_hisat2_index(genome_record.genome_sequences),
            'star_index': lambda: build_star_index(genome_record.genome_sequences, genome_record.annotation_gff),
            'minimap2_index': lambda: build_minimap2_index(genome_record.genome_sequences),
        }

        for index_name in required_index:
            if index_name in index_builders:
                existing_index = getattr(genome_record, index_name, None)
                if existing_index:
                    logger.info(f'{index_name} already exists for genome id={genome_id}, skipping.')
                    load_result[index_name] = existing_index
                else:
                    new_index = index_builders[index_name]()
                    # pyrefly: ignore [bad-argument-type]
                    genome_update = GenomeUpdate(id=genome_record.id, **{index_name: new_index})
                    genome_record = genome_service.update_genome(session, genome_update)
                    load_result[index_name] = getattr(genome_record, index_name)

    return load_result
