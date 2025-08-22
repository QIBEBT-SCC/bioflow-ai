"""
NCBI RefSeq参考基因组下载工具

使用NCBI Datasets API v2异步下载指定物种的RefSeq参考基因组及注释文件
"""

import asyncio
import aiohttp
import aiofiles
import zipfile
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import rich
from loguru import logger
from pydantic import BaseModel, Field

NCBI_API_BASE = 'https://api.ncbi.nlm.nih.gov/datasets/v2'


class NCBIOrganism(BaseModel):
    tax_id: int = Field(description='NCBI Taxonomy identifier')
    sci_name: Optional[str] = Field(None, description='Scientific name')
    organism_name: Optional[str] = Field(None, description='Scientific name')
    common_name: Optional[str] = Field(None, description='Common name')
    strain: Optional[str] = Field(None, description='Strain information')


class NCBIAssemblyInfo(BaseModel):
    assembly_level: Optional[str] = Field(None, description='The level at which a genome has been assembled')
    assembly_name: Optional[str] = Field(None, description="The assembly submitter's name for the genome assembly")
    assembly_long_name: Optional[str] = Field(None)
    release_date: Optional[str] = Field(None, description='Date the assembly was made available by NCBI')
    refseq_category: Optional[str] = Field(
        None, description='The RefSeq Category, indicates whether the assembly is a reference genome'
    )
    submitter: Optional[str] = Field(None, description='The submitting consortium or organization')


class NCBIAssemblyDataReport(BaseModel):
    accession: str = Field(description='The GenColl assembly accession')
    organism: Optional[NCBIOrganism] = Field(None, description='Organism information')
    assembly_info: Optional[NCBIAssemblyInfo] = Field(None, description='Metadata for the genome assembly submission')


class NCBIAssemblyDataReportPage(BaseModel):
    reports: list[NCBIAssemblyDataReport] = Field(default_factory=list, description='List of assembly reports')
    total_count: Optional[int] = Field(None, description='The total count of available datasets')
    next_page_token: Optional[str] = Field(None, description='Token for next page')


class NCBITaxonomyNode(BaseModel):
    tax_id: int = Field(description='NCBI Taxonomy identifier')
    organism_name: Optional[str] = Field(None, description='Scientific name')
    common_name: Optional[str] = Field(None, description='Common name')
    rank: Optional[str] = Field(None, description='Taxonomic rank')
    lineage: list[int] = Field(default_factory=list, description='Taxonomic lineage as list of tax_ids')


class NCBITaxonomyMatch(BaseModel):
    taxonomy: Optional[NCBITaxonomyNode] = Field(None, description='Taxonomy node information')


class NCBITaxonomyMetadataResponse(BaseModel):
    taxonomy_nodes: list[NCBITaxonomyMatch] = Field(default_factory=list, description='List of taxonomy matches')


class DownloadResult(BaseModel):
    species: str
    tax_id: int
    accession: str
    genome_fasta: Optional[str]
    annotation_gff: Optional[str]
    reports: Optional[str]


class NCBIReferenceDownloader:
    def __init__(self, output_dir: Optional[str] = None, timeout: int = 300):
        self.output_dir = Path(output_dir) if output_dir else Path('./genomes')
        self.timeout = timeout

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_taxonomy_info(self, tax_id: int) -> Optional[NCBITaxonomyNode]:
        """Get complete taxonomy information for a given tax_id.

        Args:
            tax_id: NCBI Taxonomy identifier

        Returns:
            NCBITaxonomyNode object containing complete taxonomy information, or None if failed
        """
        try:
            url = f'{NCBI_API_BASE}/taxonomy/taxon/{tax_id}'
            async with self.session.get(url) as response:
                response.raise_for_status()
                data = await response.json()
                taxonomy_response = NCBITaxonomyMetadataResponse.model_validate(data)
                if taxonomy_response.taxonomy_nodes and taxonomy_response.taxonomy_nodes[0].taxonomy:
                    return taxonomy_response.taxonomy_nodes[0].taxonomy
        except Exception as e:
            logger.debug(f'Failed to get taxonomy info for tax_id {tax_id}: {e}')
        return None

    async def get_species_info(self, strain_tax_id: int) -> NCBITaxonomyNode:
        """Retrieves the tax_id and name of the corresponding species based on the strain's tax_id.

        Args:
            strain_tax_id: The taxonomic ID of the strain.

        Returns:
            Tuple of (species_tax_id, species_name).
        """
        taxonomy_node = await self.get_taxonomy_info(strain_tax_id)
        if not taxonomy_node:
            logger.error(f'Failed to get taxonomy info for tax_id {strain_tax_id}, using original ID')
            raise

        # 如果当前节点就是species级别，直接返回
        if taxonomy_node.rank == 'SPECIES':
            return taxonomy_node

        # 对于非species级别，在lineage中查找species
        for lineage_tax_id in reversed(taxonomy_node.lineage):
            lineage_taxonomy = await self.get_taxonomy_info(lineage_tax_id)
            if lineage_taxonomy and lineage_taxonomy.rank == 'SPECIES':
                return lineage_taxonomy

        logger.warning(f'Species level classification not found for tax_id {strain_tax_id}, using original ID')
        return taxonomy_node

    async def search_reference_genomes(self, species_name: str) -> list[NCBIAssemblyDataReport]:
        """Searches RefSeq reference genomes by species name

        Args:
            species_name: Species name (e.g. 'Escherichia coli', 'ecoli', etc.)

        Returns:
            List of NCBI assembly data reports
        """
        logger.info(f"Searching for RefSeq reference genome for species '{species_name}'...")

        url = f'{NCBI_API_BASE}/genome/taxon/{species_name}/dataset_report'
        params = {
            'filters.reference_only': 'true',
            'filters.assembly_source': 'refseq',
            'filters.has_annotation': 'true',
            'page_size': 20,
        }

        try:
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()

                try:
                    report_page = NCBIAssemblyDataReportPage.model_validate(data)
                except Exception as e:
                    logger.error(f'Failed to parse taxonomy API response: {e}')
                    logger.debug(f'Original Response Data: {data}')
                    raise

                reports = report_page.reports
                for report in reports:
                    organism = report.organism
                    organism_name = organism.organism_name
                    logger.info(f'Find the reference genome: {report.accession} - {organism_name}')

                if not reports:
                    logger.warning(f"RefSeq reference genome for species '{species_name}' not found")

                return reports

        except aiohttp.ClientError as e:
            logger.error(f'Network error when searching genome: {e}')
            raise
        except Exception as e:
            logger.error(f'Error searching genome: {e}')
            raise

    async def download_genome_data(self, accession: str, include_annotation: bool = True) -> dict[str, str]:
        """Downloads and extracts reference genome data for a single accession.

        Args:
            accession: The genome accession number.
            include_annotation: Whether to include annotation files.

        Returns:
            Dictionary containing categorized file paths by type.
        """
        logger.info(f'Downloading and processing genome data for accession: {accession}')

        file_types: dict[str, Optional[str]] = {'genome_fasta': None, 'annotation_gff': None, 'reports': None}

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            zip_file = temp_path / f'{accession}_genome.zip'

            try:
                url = f'{NCBI_API_BASE}/genome/download'

                request_data = {'accessions': [accession]}
                if include_annotation:
                    request_data['include_annotation_type'] = [
                        'GENOME_FASTA',  # 基因组FASTA序列
                        'GENOME_GFF',  # GFF3注释文件
                        'SEQUENCE_REPORT',  # 序列报告
                    ]
                else:
                    request_data['include_annotation_type'] = ['GENOME_FASTA']

                headers = {'Content-Type': 'application/json'}

                async with self.session.post(url, json=request_data, headers=headers) as response:
                    response.raise_for_status()

                    content_type = response.headers.get('content-type', '')
                    if 'application/zip' not in content_type:
                        error_text = await response.text()
                        raise ValueError(f'The API returned a non-ZIP format response: {error_text}')

                    async with aiofiles.open(zip_file, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            await f.write(chunk)

                logger.info('Downloaded genome data to temporary location')

                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(temp_path)

                data_dir = temp_path / 'ncbi_dataset' / 'data'
                if not data_dir.exists():
                    logger.warning('ncbi_dataset/data directory not found')
                    raise

                accession_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith(('GCF_', 'GCA_'))]
                if not accession_dirs:
                    logger.warning('No genome accession directory found')
                    raise

                assert len(accession_dirs) == 1, (
                    f'Expected exactly 1 accession directory, found {len(accession_dirs)}: {[d.name for d in accession_dirs]}'
                )

                accession_dir = accession_dirs[0]
                found_accession = accession_dir.name
                logger.info(f'Processing genome: {found_accession}')

                target_dir = self.output_dir / found_accession
                target_dir.mkdir(exist_ok=True)

                for item in accession_dir.iterdir():
                    if item.is_file():
                        target_file = target_dir / item.name
                        shutil.copy2(item, target_file)

                        file_lower = item.name.lower()

                        if file_lower.endswith('.fna') or file_lower.endswith('.fasta'):
                            if 'genomic' in file_lower:
                                file_types['genome_fasta'] = str(target_file)
                        elif file_lower.endswith('.gff') or file_lower.endswith('.gff3'):
                            file_types['annotation_gff'] = str(target_file)
                        elif file_lower.endswith('.jsonl') or file_lower.endswith('.tsv'):
                            file_types['reports'] = str(target_file)

                logger.info(f'Successfully copied {found_accession} to: {target_dir}')
                return file_types

            except aiohttp.ClientError as e:
                logger.error(f'Network error when downloading genome data: {e}')
                raise
            except Exception as e:
                logger.error(f'Error downloading genome data: {e}')
                raise

    @staticmethod
    def select_best_genome(genomes: list[NCBIAssemblyDataReport]) -> NCBIAssemblyDataReport:
        """Select the best reference genome from multiple reference genomes

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
            raise ValueError('No selectable reference genome')

        if len(genomes) == 1:
            return genomes[0]

        logger.info(f'Selecting the best of {len(genomes)} genomes...')

        refseq_priority = {'reference genome': 3, 'representative genome': 2, 'na': 1}
        level_priority = {'complete genome': 4, 'chromosome': 3, 'scaffold': 2, 'contig': 1}

        def genome_score(_genome: NCBIAssemblyDataReport) -> tuple:
            """计算基因组得分，返回用于排序的元组"""
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


async def download_reference_genome(species_name: str, output_dir: str = './genome_data') -> DownloadResult:
    """Download RefSeq reference genome by species name

    Args:
        species_name: Species name (e.g., ‘Escherichia coli’, ‘ecoli’)
        output_dir: Output directory

    Returns:
        DownloadResult struct containing information about the downloaded genome

    Note:
        This function guarantees to return only one best reference genome
    """

    try:
        async with NCBIReferenceDownloader(output_dir) as downloader:
            # 1. 搜索参考基因组
            genomes = await downloader.search_reference_genomes(species_name)

            if not genomes:
                raise ValueError(f"RefSeq reference genome for species '{species_name}' not found")

            # 2. 选择最佳的基因组(保证唯一性)
            best_genome = downloader.select_best_genome(genomes)

            # 3. 下载基因组数据
            extracted_files = await downloader.download_genome_data(best_genome.accession, include_annotation=True)

            # 4. 获取基因组信息
            best_organism = best_genome.organism
            organism_name = best_organism.organism_name or best_organism.sci_name

            logger.info(f'Genome downloaded successfully: {best_genome.accession} - {organism_name}')

            # 5. 获取物种级别的分类信息
            species_info = await downloader.get_species_info(best_organism.tax_id)

            return DownloadResult(
                species=species_info.organism_name,
                tax_id=species_info.tax_id,
                accession=best_genome.accession,
                **extracted_files,
            )

    except Exception as e:
        error_msg = f'下载过程中发生错误: {e}'
        logger.error(error_msg)
        raise


async def main():
    # 下载大肠杆菌参考基因组
    result = await download_reference_genome('E coli', './ecoli_genome')
    rich.print(result)


if __name__ == '__main__':
    asyncio.run(main())
