from pathlib import Path
from typing import Any, Optional

import rich
from celery import Task
from docker.types import Mount
from loguru import logger
from sqlmodel import Session

import app.crud.resource.reference_genome as genome_crud
from app.core.config import get_settings
from app.core.session import engine
from app.models.resource import ReferenceGenomes
from app.models.tools import ImageConfig
from app.models.workflow import ToolOutput
from app.schemas.react_flow import TaskInput
from app.schemas.resource import GenomeUpdate
from app.tasks.base.base_task import BaseBioTask, get_tool_output
from app.utils.api.ncbi_datasets import (
    download_genome_data,
    get_species_info,
    search_reference_genomes,
    search_reference_genomes_by_tax_id,
)
from worker import celery_app


def _run_command(image_cfg: ImageConfig, mounts: list[Mount], command: str):
    if get_settings().docker.CONTAINER == 'podman':
        from app.utils.container.podman import run_podman

        status, output = run_podman(
            image_cfg,
            command,
            mounts,
        )
    else:
        from app.utils.container.docker import run_docker

        status, output = run_docker(
            image_cfg,
            command,
            mounts,
        )

    return status, output


def build_bowtie2_index(genome_fasta: str) -> str:
    genome_fasta = Path(genome_fasta)
    index_path = genome_fasta.parent / 'bowtie2'
    index_path.mkdir(exist_ok=True, parents=True)

    mounts = [
        Mount(type='bind', source=str(genome_fasta.parent), target='/data/raw'),
        Mount(type='bind', source=str(index_path), target='/data/output'),
    ]

    image_cfg = ImageConfig.from_str('quay.io/biocontainers/bowtie2:2.5.4--he96a11b_6')
    status, output = _run_command(
        image_cfg,
        mounts,
        f'bowtie2-build --threads {get_settings().persistent.MAX_THREAD} /data/raw/{genome_fasta.name} /data/output/{genome_fasta.stem}',
    )

    assert status == 0
    return f'{index_path}/{genome_fasta.stem}'


def build_bwa_index(genome_fasta: str):
    genome_fasta = Path(genome_fasta)

    mounts = [
        Mount(type='bind', source=str(genome_fasta.parent), target='/data/raw'),
    ]

    image_cfg = ImageConfig.from_str('quay.io/biocontainers/bwa:0.7.19--h577a1d6_1')
    status, output = _run_command(
        image_cfg,
        mounts,
        f'bwa index /data/raw/{genome_fasta.name}',
    )

    assert status == 0
    return f'{genome_fasta}'


def build_hisat2_index(genome_fasta: str):
    genome_fasta = Path(genome_fasta)
    index_path = genome_fasta.parent / 'hisat2'
    index_path.mkdir(exist_ok=True, parents=True)

    mounts = [
        Mount(type='bind', source=str(genome_fasta.parent), target='/data/raw'),
        Mount(type='bind', source=str(index_path), target='/data/output'),
    ]

    image_cfg = ImageConfig.from_str('quay.io/biocontainers/hisat2:2.2.1--h503566f_8')
    status, output = _run_command(
        image_cfg,
        mounts,
        f'hisat2-build -p {get_settings().persistent.MAX_THREAD} /data/raw/{genome_fasta.name} /data/output/{genome_fasta.stem}',
    )

    if status == 0:
        logger.success('build hisat2 index succeed')
    else:
        logger.error(output)
        raise
    return f'{index_path}/{genome_fasta.stem}'


def build_star_index(genome_fasta: str, annotation_gff: str):
    genome_fasta = Path(genome_fasta)
    annotation_gff = Path(annotation_gff)
    index_path = genome_fasta.parent / 'star'
    index_path.mkdir(exist_ok=True, parents=True)

    mounts = [
        Mount(type='bind', source=str(genome_fasta.parent), target='/data/raw'),
        Mount(type='bind', source=str(index_path), target='/data/output'),
    ]

    image_cfg = ImageConfig.from_str('quay.io/biocontainers/star:2.7.11b--h5ca1c30_7')
    status, output = _run_command(
        image_cfg,
        mounts,
        f'STAR --runThreadN {get_settings().persistent.MAX_THREAD} --runMode genomeGenerate --genomeDir /data/output --genomeFastaFiles /data/raw/{genome_fasta.name} --sjdbGTFfile /data/raw/{annotation_gff.name}',
    )

    assert status == 0
    return f'{index_path}'


def build_minimap2_index(genome_fasta: str):
    genome_fasta = Path(genome_fasta)
    index_path = genome_fasta.parent / 'minimap2'
    index_path.mkdir(exist_ok=True, parents=True)

    mounts = [
        Mount(type='bind', source=str(genome_fasta.parent), target='/data/raw'),
        Mount(type='bind', source=str(index_path), target='/data/output'),
    ]

    image_cfg = ImageConfig.from_str('quay.io/biocontainers/minimap2:2.30--h577a1d6_0')
    status, output = _run_command(
        image_cfg,
        mounts,
        f'minimap2 -t {get_settings().persistent.MAX_THREAD} -d /data/output/{genome_fasta.stem}.mmi /data/raw/{genome_fasta.name} ',
    )

    assert status == 0
    return f'{index_path}/{genome_fasta.stem}.mmi'


def _create_genome_record(
    session: Session, genome, species_info: Any, alias_name: Optional[str] = None
) -> ReferenceGenomes:
    """创建基因组数据库记录"""
    download_result = download_genome_data(
        genome.accession, get_settings().persistent.GENOME_PATH, include_annotation=True, timeout=3600
    )

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

    return genome_crud.insert(session, genome_record)


@celery_app.task(bind=True, pydantic=True, base=BaseBioTask)
def load_genome_task(
    self: Task, result_dict: Optional[dict[str, Any]] = None, *, inputs: dict[str, TaskInput], required_index: list[str]
) -> ToolOutput:
    load_result = {}
    with Session(engine) as session:
        if 'ncbi_tax_id' in inputs:
            tax_id = get_tool_output(inputs['ncbi_tax_id'].source_uid, inputs['ncbi_tax_id'].source_k)
            # 首先尝试通过tax_id在数据库中查找
            genome_record = genome_crud.get_by_taxid(session, int(tax_id))

            if not genome_record:
                # 如果数据库中没有，通过API搜索并下载
                genome = search_reference_genomes_by_tax_id(int(tax_id))
                species_info = get_species_info(genome.organism.tax_id)
                genome_record = _create_genome_record(session, genome, species_info)
            else:
                logger.info(f'Found existed genome record for tax_id: "{tax_id}"')
        elif 'species_name' in inputs:
            species_name = get_tool_output(inputs['species_name'].source_uid, inputs['species_name'].source_k)
            genome_records = genome_crud.search(session, species_name)

            if not genome_records:
                genome = search_reference_genomes(species_name)
                species_info = get_species_info(genome.organism.tax_id)
                rich.print(species_info)

                if genome_record := genome_crud.get_by_taxid(session, species_info.tax_id):
                    genome_update = GenomeUpdate(id=genome_record.id, aliases=f'{genome_record.aliases}|{species_name}')
                    genome_record = genome_crud.update(session, genome_update)
                else:
                    genome_record = _create_genome_record(session, genome, species_info, species_name)
            else:
                logger.info(f'Found existed genome record for species: "{species_name}"')
                genome_record = genome_records[0]

        else:
            raise ValueError("Either 'ncbi_tax_id' or 'species_name' must be provided in inputs")

        load_result['genome_fasta'] = genome_record.genome_sequences
        load_result['annotation_gff'] = genome_record.annotation_gff
        load_result['annotation_gtf'] = genome_record.annotation_gtf

        # 定义索引构建器映射
        index_builders = {
            'bowtie2_index': lambda: build_bowtie2_index(genome_record.genome_sequences),
            'bwa_index': lambda: build_bwa_index(genome_record.genome_sequences),
            'hisat2_index': lambda: build_hisat2_index(genome_record.genome_sequences),
            'star_index': lambda: build_star_index(genome_record.genome_sequences, genome_record.annotation_features),
            'minimap2_index': lambda: build_minimap2_index(genome_record.genome_sequences),
        }

        # 处理每个需要的索引
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
                    genome_update = GenomeUpdate(id=genome_record.id, **update_data)
                    genome_record = genome_crud.update(session, genome_update)
                    load_result[index_name] = getattr(genome_record, index_name)

    return ToolOutput(result=load_result)
