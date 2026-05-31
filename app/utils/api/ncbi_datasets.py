"""NCBI RefSeq Reference Genome Download Tool.

Download RefSeq reference genomes and annotation files for specified species using the NCBI Datasets API v2.
Supports both synchronous and asynchronous methods, utilizing httpx as the HTTP client.
"""

import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Literal

import httpx
from loguru import logger
from pydantic import BaseModel, Field

from app.core.config import get_settings

NCBI_API_BASE = 'https://api.ncbi.nlm.nih.gov/datasets/v2'


class NCBIOrganism(BaseModel):
    tax_id: int = Field(description='NCBI Taxonomy identifier')
    sci_name: str | None = Field(None, description='Scientific name')
    organism_name: str | None = Field(None, description='Scientific name')
    common_name: str | None = Field(None, description='Common name')
    strain: str | None = Field(None, description='Strain information')


class NCBIAssemblyInfo(BaseModel):
    assembly_level: str | None = Field(None, description='The level at which a genome has been assembled')
    assembly_name: str | None = Field(None, description="The assembly submitter's name for the genome assembly")
    assembly_long_name: str | None = Field(None)
    release_date: str | None = Field(None, description='Date the assembly was made available by NCBI')
    refseq_category: str | None = Field(
        None, description='The RefSeq Category, indicates whether the assembly is a reference genome'
    )
    submitter: str | None = Field(None, description='The submitting consortium or organization')


class NCBIAssemblyDataReport(BaseModel):
    accession: str = Field(description='The GenColl assembly accession')
    organism: NCBIOrganism = Field(description='Organism information')
    assembly_info: NCBIAssemblyInfo = Field(description='Metadata for the genome assembly submission')


class NCBIAssemblyDataReportPage(BaseModel):
    reports: list[NCBIAssemblyDataReport] = Field(default_factory=list, description='List of assembly reports')
    total_count: int | None = Field(None, description='The total count of available datasets')
    next_page_token: str | None = Field(None, description='Token for next page')


class NCBITaxonomyNode(BaseModel):
    tax_id: int = Field(description='NCBI Taxonomy identifier')
    organism_name: str = Field(description='Scientific name')
    common_name: str | None = Field(None, description='Common name')
    rank: str | None = Field(None, description='Taxonomic rank')
    lineage: list[int] = Field(default_factory=list, description='Taxonomic lineage as list of tax_ids')


class NCBITaxonomyMatch(BaseModel):
    taxonomy: NCBITaxonomyNode | None = Field(None, description='Taxonomy node information')


class NCBITaxonomyMetadataResponse(BaseModel):
    taxonomy_nodes: list[NCBITaxonomyMatch] = Field(default_factory=list, description='List of taxonomy matches')


class DownloadResult(BaseModel):
    species: str
    tax_id: int
    accession: str
    genome_fasta: str | None
    annotation_gff: str | None
    reports: str | None


def _get_http_client(timeout: int = 300) -> httpx.Client:
    proxy = get_settings().network.get_proxy_url() if get_settings().network.USE_PROXY else None
    return httpx.Client(timeout=timeout, proxy=proxy)


def _get_async_http_client(timeout: int = 300) -> httpx.AsyncClient:
    proxy = get_settings().network.get_proxy_url() if get_settings().network.USE_PROXY else None
    return httpx.AsyncClient(timeout=timeout, proxy=proxy)


def _make_request(
    client: httpx.Client,
    method: str,
    url: str,
    **kwargs,
) -> httpx.Response:
    match method.upper():
        case 'GET':
            return client.get(url, **kwargs)
        case 'POST':
            return client.post(url, **kwargs)
        case _:
            err_msg = f'Unsupported HTTP method: {method}'
            raise ValueError(err_msg)


async def _make_request_async(
    client: httpx.AsyncClient,
    method: Literal['GET', 'POST'],
    url: str,
    **kwargs,
) -> httpx.Response:
    match method.upper():
        case 'GET':
            return await client.get(url, **kwargs)
        case 'POST':
            return await client.post(url, **kwargs)
        case _:
            err_msg = f'Unsupported HTTP method: {method}'
            raise ValueError(err_msg)


def _handle_response(response: httpx.Response, expected_content_type: str | None = None) -> Any:
    response.raise_for_status()

    if expected_content_type:
        content_type = response.headers.get('content-type', '')
        if expected_content_type not in content_type:
            err_msg = f'The API returned an unexpected content type: {content_type}, response: {response.text}'
            raise ValueError(err_msg)

    return response.json() if 'application/json' in response.headers.get('content-type', '') else response


def _select_best_genome(genomes: list[NCBIAssemblyDataReport]) -> NCBIAssemblyDataReport:
    """Select the best reference genome from multiple reference genomes.

    Selection priority:
    1. Reference genome > Representative genome > Others
    2. Assembly level: Complete Genome > Chromosome > Scaffold > Contig
    3. Latest release date

    Args:
        genomes: List of NCBI reference genomes

    Returns:
        The best reference genome
    """
    if not genomes:
        err_msg = 'No selectable reference genome'
        raise ValueError(err_msg)

    if len(genomes) == 1:
        return genomes[0]

    logger.info(f'Selecting the best of {len(genomes)} genomes...')

    refseq_priority = {'reference genome': 3, 'representative genome': 2, 'na': 1}
    level_priority = {'complete genome': 4, 'chromosome': 3, 'scaffold': 2, 'contig': 1}

    def genome_score(_genome: NCBIAssemblyDataReport) -> tuple:
        """Computing the genome sorting score."""
        _assembly_info = _genome.assembly_info

        # RefSeq category
        refseq_category = (_assembly_info.refseq_category or '').lower()
        refseq_score = refseq_priority.get(refseq_category, 0)

        # Assembly level
        _assembly_level = (_assembly_info.assembly_level or '').lower()
        level_score = level_priority.get(_assembly_level, 0)

        # Release date
        date_score = _assembly_info.release_date or ''

        return refseq_score, level_score, date_score

    # 排序并选择最佳的
    sorted_genomes = sorted(genomes, key=genome_score, reverse=True)
    best_genome = sorted_genomes[0]

    best_organism = best_genome.organism
    best_assembly = best_genome.assembly_info

    logger.info('The Best Genome Choice:')
    logger.info(f'  Accession: {best_genome.accession}')
    logger.info(f'  Organism: {best_organism.organism_name or best_organism.sci_name}')
    logger.info(f'  RefSeq Category: {best_assembly.refseq_category}')
    logger.info(f'  Assembly Level: {best_assembly.assembly_level}')
    logger.info(f'  Release Date: {best_assembly.release_date}')

    if len(genomes) > 1:
        logger.info(f'Other {len(genomes) - 1} genomes skipped:')
        for genome in sorted_genomes[1:]:
            assembly_info = genome.assembly_info
            logger.info(f'  - {genome.accession} ({assembly_info.refseq_category}, {assembly_info.assembly_level})')

    return best_genome


def _process_zip_file(zip_file: Path, temp_path: Path, output_dir: Path) -> dict[str, str]:
    file_types: dict[str, str] = {}

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(temp_path)

    data_dir = temp_path / 'ncbi_dataset' / 'data'
    if not data_dir.exists():
        logger.warning('ncbi_dataset/data directory not found')
        err_msg = 'Invalid ZIP file structure'
        raise ValueError(err_msg)

    accession_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith(('GCF_', 'GCA_'))]
    if not accession_dirs:
        logger.warning('No genome accession directory found')
        err_msg = 'No genome accession directory found'
        raise ValueError(err_msg)

    accession_dir = accession_dirs[0]
    found_accession = accession_dir.name
    logger.info(f'Processing genome: {found_accession}')

    target_dir = output_dir / found_accession
    target_dir.mkdir(exist_ok=True)

    for item in accession_dir.iterdir():
        if item.is_file():
            target_file = target_dir / item.name
            shutil.copy2(item, target_file)

            file_lower = item.name.lower()

            if file_lower.endswith(('.fna', '.fasta')):
                if 'genomic' in file_lower:
                    file_types['genome_fasta'] = str(target_file.absolute())
            elif file_lower.endswith(('.gff', '.gff3')):
                file_types['annotation_gff'] = str(target_file.absolute())
            elif file_lower.endswith('.gtf'):
                file_types['annotation_gtf'] = str(target_file.absolute())
            elif file_lower.endswith(('.jsonl', '.tsv')):
                file_types['reports'] = str(target_file.absolute())

    logger.info(f'Successfully copied {found_accession} to: {target_dir}')
    return file_types


def get_taxonomy_info(tax_id: int, timeout: int = 300) -> NCBITaxonomyNode | None:
    """Get complete taxonomy information for a given tax_id (synchronous version)."""
    try:
        with _get_http_client(timeout) as client:
            url = f'{NCBI_API_BASE}/taxonomy/taxon/{tax_id}'
            response = _make_request(client, 'GET', url)
            data = _handle_response(response)
            taxonomy_response = NCBITaxonomyMetadataResponse.model_validate(data)
            if taxonomy_response.taxonomy_nodes and taxonomy_response.taxonomy_nodes[0].taxonomy:
                return taxonomy_response.taxonomy_nodes[0].taxonomy
    except Exception as e:
        logger.debug(f'Failed to get taxonomy info for tax_id {tax_id}: {e}')
    return None


async def aget_taxonomy_info(tax_id: int, timeout: int = 300) -> NCBITaxonomyNode | None:
    """Get complete taxonomy information for a given tax_id (asynchronous version)."""
    try:
        async with _get_async_http_client(timeout) as client:
            url = f'{NCBI_API_BASE}/taxonomy/taxon/{tax_id}'
            response = await _make_request_async(client, 'GET', url)
            data = _handle_response(response)
            taxonomy_response = NCBITaxonomyMetadataResponse.model_validate(data)
            if taxonomy_response.taxonomy_nodes and taxonomy_response.taxonomy_nodes[0].taxonomy:
                return taxonomy_response.taxonomy_nodes[0].taxonomy
    except Exception as e:
        logger.debug(f'Failed to get taxonomy info for tax_id {tax_id}: {e}')
    return None


def get_species_info(strain_tax_id: int, timeout: int = 300) -> NCBITaxonomyNode:
    """Retrieves the tax_id and name of the corresponding species based on the strain's tax_id (synchronous version)."""
    taxonomy_node = get_taxonomy_info(strain_tax_id, timeout)

    if not taxonomy_node:
        logger.error(f'Failed to get taxonomy info for tax_id {strain_tax_id}, using original ID')
        raise ValueError

    # 如果当前节点就是species级别, 直接返回
    if taxonomy_node.rank == 'SPECIES':
        return taxonomy_node

    # 对于非species级别, 在lineage中查找species
    for lineage_tax_id in reversed(taxonomy_node.lineage):
        lineage_taxonomy = get_taxonomy_info(lineage_tax_id, timeout)

        if lineage_taxonomy and lineage_taxonomy.rank == 'SPECIES':
            return lineage_taxonomy

    logger.warning(f'Species level classification not found for tax_id {strain_tax_id}, using original ID')
    return taxonomy_node


async def aget_species_info(strain_tax_id: int, timeout: int = 300) -> NCBITaxonomyNode:
    """Retrieves the tax_id and name of the corresponding species based on the strain's tax_id (asynchronous version)."""
    taxonomy_node = await aget_taxonomy_info(strain_tax_id, timeout)

    if not taxonomy_node:
        logger.error(f'Failed to get taxonomy info for tax_id {strain_tax_id}, using original ID')
        raise ValueError

    if taxonomy_node.rank == 'SPECIES':
        return taxonomy_node

    for lineage_tax_id in reversed(taxonomy_node.lineage):
        lineage_taxonomy = await aget_taxonomy_info(lineage_tax_id, timeout)

        if lineage_taxonomy and lineage_taxonomy.rank == 'SPECIES':
            return lineage_taxonomy

    logger.warning(f'Species level classification not found for tax_id {strain_tax_id}, using original ID')
    return taxonomy_node


def search_reference_genomes(species_name: str, timeout: int = 300) -> NCBIAssemblyDataReport:
    """Searches RefSeq reference genomes by species name."""
    logger.info(f"Searching for RefSeq reference genome for species '{species_name}'...")

    url = f'{NCBI_API_BASE}/genome/taxon/{species_name}/dataset_report'
    params = {
        'filters.reference_only': 'true',
        'filters.assembly_source': 'refseq',
        'filters.has_annotation': 'true',
        'page_size': 20,
    }

    try:
        with _get_http_client(timeout) as client:
            response = _make_request(client, 'GET', url, params=params)
            data = _handle_response(response)

            try:
                report_page = NCBIAssemblyDataReportPage.model_validate(data)
            except Exception as e:
                logger.error(f'Failed to parse taxonomy API response: {e}')
                logger.debug(f'Original Response Data: {data}')
                raise

            reports = report_page.reports

            if not reports:
                logger.warning(f"RefSeq reference genome for species '{species_name}' not found")

            return _select_best_genome(reports)

    except httpx.HTTPError as e:
        logger.error(f'Network error when searching genome: {e}')
        raise
    except Exception as e:
        logger.exception(f'Error searching genome: {e}')
        raise


async def asearch_reference_genomes(species_name: str, timeout: int = 300) -> NCBIAssemblyDataReport:
    """Searches RefSeq reference genomes by species name."""
    logger.info(f"Searching for RefSeq reference genome for species '{species_name}'...")

    url = f'{NCBI_API_BASE}/genome/taxon/{species_name}/dataset_report'
    params = {
        'filters.reference_only': 'true',
        'filters.assembly_source': 'refseq',
        'filters.has_annotation': 'true',
        'page_size': 20,
    }

    try:
        async with _get_async_http_client(timeout) as client:
            response = await _make_request_async(client, 'GET', url, params=params)
            data = _handle_response(response)

            try:
                report_page = NCBIAssemblyDataReportPage.model_validate(data)
            except Exception as e:
                logger.error(f'Failed to parse taxonomy API response: {e}')
                logger.debug(f'Original Response Data: {data}')
                raise

            reports = report_page.reports

            if not reports:
                logger.warning(f"RefSeq reference genome for species '{species_name}' not found")

            return _select_best_genome(reports)

    except httpx.HTTPError as e:
        logger.error(f'Network error when searching genome: {e}')
        raise
    except Exception as e:
        logger.exception(f'Error searching genome: {e}')
        raise


def search_reference_genomes_by_tax_id(tax_id: int, timeout: int = 300) -> NCBIAssemblyDataReport:
    """Searches RefSeq reference genomes by NCBI taxonomy ID."""
    logger.info(f"Searching for RefSeq reference genome for tax_id '{tax_id}'...")

    url = f'{NCBI_API_BASE}/genome/taxon/{tax_id}/dataset_report'
    params = {
        'filters.reference_only': 'true',
        'filters.assembly_source': 'refseq',
        'filters.has_annotation': 'true',
        'page_size': 20,
    }

    try:
        with _get_http_client(timeout) as client:
            response = _make_request(client, 'GET', url, params=params)
            data = _handle_response(response)

            try:
                report_page = NCBIAssemblyDataReportPage.model_validate(data)
            except Exception as e:
                logger.error(f'Failed to parse taxonomy API response: {e}')
                logger.debug(f'Original Response Data: {data}')
                raise

            reports = report_page.reports

            if not reports:
                logger.warning(f"RefSeq reference genome for tax_id '{tax_id}' not found")

            return _select_best_genome(reports)

    except httpx.HTTPError as e:
        logger.error(f'Network error when searching genome: {e}')
        raise
    except Exception as e:
        logger.exception(f'Error searching genome: {e}')
        raise


async def asearch_reference_genomes_by_tax_id(tax_id: int, timeout: int = 300) -> NCBIAssemblyDataReport:
    """Searches RefSeq reference genomes by NCBI taxonomy ID."""
    logger.info(f"Searching for RefSeq reference genome for tax_id '{tax_id}'...")

    url = f'{NCBI_API_BASE}/genome/taxon/{tax_id}/dataset_report'
    params = {
        'filters.reference_only': 'true',
        'filters.assembly_source': 'refseq',
        'filters.has_annotation': 'true',
        'page_size': 20,
    }

    try:
        async with _get_async_http_client(timeout) as client:
            response = await _make_request_async(client, 'GET', url, params=params)
            data = _handle_response(response)

            try:
                report_page = NCBIAssemblyDataReportPage.model_validate(data)
            except Exception as e:
                logger.error(f'Failed to parse taxonomy API response: {e}')
                logger.debug(f'Original Response Data: {data}')
                raise

            reports = report_page.reports

            if not reports:
                logger.warning(f"RefSeq reference genome for tax_id '{tax_id}' not found")

            return _select_best_genome(reports)

    except httpx.HTTPError as e:
        logger.error(f'Network error when searching genome: {e}')
        raise
    except Exception as e:
        logger.exception(f'Error searching genome: {e}')
        raise


def download_genome_data(
    accession: str, output_dir: str = './genomes', *, include_annotation: bool = True, timeout: int = 300
) -> dict[str, str]:
    """Downloads and extracts reference genome data for a single accession."""
    logger.info(f'Downloading and processing genome data for accession: {accession}')

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        zip_file = temp_path / f'{accession}_genome.zip'

        try:
            with _get_http_client(timeout) as client:
                url = f'{NCBI_API_BASE}/genome/download'

                request_data = {'accessions': [accession]}
                if include_annotation:
                    request_data['include_annotation_type'] = [
                        'GENOME_FASTA',  # 基因组FASTA序列
                        'GENOME_GFF',  # GFF3注释文件
                        'GENOME_GTF',  # GTF注释文件
                        'SEQUENCE_REPORT',  # 序列报告
                    ]
                else:
                    request_data['include_annotation_type'] = ['GENOME_FASTA']

                headers = {'Content-Type': 'application/json'}

                response = _make_request(client, 'POST', url, json=request_data, headers=headers)

                # 检查响应类型
                content_type = response.headers.get('content-type', '')
                if 'application/zip' not in content_type:
                    error_text = f'The API returned a non-ZIP format response: {response.text}'
                    raise ValueError(error_text)  # noqa: TRY301

                # 下载文件
                with zip_file.open('wb') as f:
                    for chunk in response.iter_bytes(chunk_size=8192):
                        f.write(chunk)

                logger.info('Downloaded genome data to temporary location')

                # 处理ZIP文件
                return _process_zip_file(zip_file, temp_path, output_path)

        except httpx.HTTPError as e:
            logger.error(f'Network error when downloading genome data: {e}')
            raise
        except Exception as e:
            logger.error(f'Error downloading genome data: {e}')
            raise


async def adownload_genome_data(
    accession: str, output_dir: str = './genomes', *, include_annotation: bool = True, timeout: int = 300
) -> dict[str, str]:
    """Downloads and extracts reference genome data for a single accession (asynchronous version)."""
    logger.info(f'Downloading and processing genome data for accession: {accession}')

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        zip_file = temp_path / f'{accession}_genome.zip'

        try:
            async with _get_async_http_client(timeout) as client:
                url = f'{NCBI_API_BASE}/genome/download'

                request_data = {'accessions': [accession]}
                if include_annotation:
                    request_data['include_annotation_type'] = [
                        'GENOME_FASTA',
                        'GENOME_GFF',
                        'SEQUENCE_REPORT',
                    ]
                else:
                    request_data['include_annotation_type'] = ['GENOME_FASTA']

                headers = {'Content-Type': 'application/json'}

                response = await _make_request_async(client, 'POST', url, json=request_data, headers=headers)
                content_type = response.headers.get('content-type', '')
                if 'application/zip' not in content_type:
                    error_text = response.text
                    logger.error(f'The API returned a non-ZIP format response: {error_text}')

                # 下载文件
                with zip_file.open('wb') as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        f.write(chunk)

                logger.info('Downloaded genome data to temporary location')

                # 处理ZIP文件
                return _process_zip_file(zip_file, temp_path, output_path)

        except httpx.HTTPError as e:
            logger.error(f'Network error when downloading genome data: {e}')
            raise
        except Exception as e:
            logger.error(f'Error downloading genome data: {e}')
            raise


def download_reference_genome(
    species_name: str, output_dir: str = './genome_data', timeout: int = 300
) -> DownloadResult:
    """Download RefSeq reference genome by species name."""
    try:
        # 1. 搜索参考基因组
        genome = search_reference_genomes(species_name, timeout)
        # 2. 获取物种级别的分类信息
        species_info = get_species_info(genome.organism.tax_id, timeout)
        # 3. 下载基因组数据
        extracted_files = download_genome_data(genome.accession, output_dir, include_annotation=True, timeout=timeout)
        logger.info(f'Genome downloaded successfully: {genome.accession}')

        return DownloadResult(
            species=species_info.organism_name,
            tax_id=species_info.tax_id,
            accession=genome.accession,
            **extracted_files,
        )

    except Exception as e:
        error_msg = f'下载过程中发生错误: {e}'
        logger.error(error_msg)
        raise


async def adownload_reference_genome(
    species_name: str, output_dir: str = './genome_data', timeout: int = 300
) -> DownloadResult:
    """Download RefSeq reference genome by species name (asynchronous version)."""
    try:
        # 1. 搜索参考基因组
        genome = await asearch_reference_genomes(species_name, timeout)

        # 3. 获取物种级别的分类信息
        species_info = await aget_species_info(genome.organism.tax_id, timeout)

        # 4. 下载基因组数据
        extracted_files = await adownload_genome_data(
            genome.accession, output_dir, include_annotation=True, timeout=timeout
        )
        logger.info(f'Genome downloaded successfully: {genome.accession}')

        return DownloadResult(
            species=species_info.organism_name,
            tax_id=species_info.tax_id,
            accession=genome.accession,
            **extracted_files,
        )

    except Exception as e:
        error_msg = f'下载过程中发生错误: {e}'
        logger.error(error_msg)
        raise
