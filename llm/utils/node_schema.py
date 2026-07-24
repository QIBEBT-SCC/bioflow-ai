from collections.abc import Callable
from pathlib import Path
from typing import NotRequired, TypedDict

from pydantic import BaseModel, field_validator

from app.schemas.tools import Handle

# Forward declaration for type hint
from llm.states import BuildEdge


class NodeSchema(TypedDict):
    name: str
    description: NotRequired[str]
    args: type[BaseModel]
    input_handles: list[Handle]
    output_handles: list[Handle]
    edge_check: NotRequired[Callable[[list[BuildEdge], list[Handle]], str | None]]


class SimpleSchema(BaseModel):
    pass


# 处理器节点
class GlobalMarkSchema(BaseModel):
    mark_name: str
    description: str


class SelectFileInFolderSchema(BaseModel):
    file_name: str


class MoveFilesToFolderSchema(BaseModel):
    folder_name: str


class RenameFileSchema(BaseModel):
    new_file_name: str

    @field_validator('new_file_name')
    @classmethod
    def validate_new_file_name(cls, new_file_name: str) -> str:
        """Require a plain file name rather than a path."""
        if (
            not new_file_name
            or new_file_name in {'.', '..'}
            or Path(new_file_name).name != new_file_name
            or '\\' in new_file_name
        ):
            msg = 'new_file_name must be a single, non-empty file name'
            raise ValueError(msg)
        return new_file_name


class CompressSchema(BaseModel):
    compressed_file_name: str

    @field_validator('compressed_file_name')
    @classmethod
    def validate_compressed_file_name(cls, compressed_file_name: str) -> str:
        """Validate the output name and use its suffix to select the compression format."""
        if (
            not compressed_file_name
            or compressed_file_name in {'.', '..'}
            or Path(compressed_file_name).name != compressed_file_name
            or '\\' in compressed_file_name
        ):
            msg = 'compressed_file_name must be a single, non-empty file name'
            raise ValueError(msg)

        supported_suffixes = {'.gz', '.bz2', '.xz', '.zip', '.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2', '.tar.xz', '.txz'}
        if not any(compressed_file_name.endswith(suffix) for suffix in supported_suffixes):
            supported = ', '.join(sorted(supported_suffixes))
            msg = f'unsupported compression suffix; supported suffixes: {supported}'
            raise ValueError(msg)
        return compressed_file_name


class BindParamSchema(BaseModel):
    parameter: str


# 资源节点
class DbResourceSchema(BaseModel):
    db_id: int


class ValueStringResourceSchema(BaseModel):
    value: str


class SequenceResourceSchema(BaseModel):
    r1: str
    r2: str


class FileInputSchema(BaseModel):
    file_path: str


class TextFileResourceSchema(BaseModel):
    file_name: str
    content: str

    @field_validator('file_name')
    @classmethod
    def validate_file_name(cls, file_name: str) -> str:
        """Require a plain file name rather than a path."""
        if not file_name or file_name in {'.', '..'} or Path(file_name).name != file_name or '\\' in file_name:
            msg = 'file_name must be a single, non-empty file name'
            raise ValueError(msg)
        return file_name

    @field_validator('content')
    @classmethod
    def validate_content_size(cls, content: str) -> str:
        """Keep inline text resources small enough for workflow definitions."""
        max_bytes = 5 * 1024 * 1024
        if len(content.encode('utf-8')) > max_bytes:
            msg = f'content must not exceed {max_bytes} UTF-8 bytes'
            raise ValueError(msg)
        return content


PROCESSORS: dict[str, NodeSchema] = {
    'global_mark': {
        'name': 'Global Mark',
        'description': 'Registers an upstream file or directory as a project-level path. Other workflows load it through the general `file` resource with `file_path="proj:<mark_name>"`. `mark_name` is the unique project keyword without the `proj:` prefix, and `description` explains the path for later workflow design.',
        'args': GlobalMarkSchema,
        'input_handles': [Handle(name='file', description='File or directory to register for cross-workflow use')],
        'output_handles': [],
    },
    'bind_param': {
        'name': 'Bind Parameter',
        'description': 'This node binds a file from another node to a specified parameter name for use by the CODE node. Its output can only be used as input for a CODE node.',
        'args': BindParamSchema,
        'input_handles': [Handle(name='file', description='File corresponding to the parameter')],
        'output_handles': [Handle(name='file_with_param', description='Available for code node input')],
    },
    'select_file_in_folder': {
        'name': 'Select File in Folder',
        'description': 'This node is used to select a specific file from a folder. Connect the input handle of this node to the output handle representing the folder. Use the `file_name` parameter to specify the name of the file you want to select within the folder. The output handle of this node will provide the path to the selected file.',
        'args': SelectFileInFolderSchema,
        'input_handles': [Handle(name='folder_path', description='Folder path')],
        'output_handles': [Handle(name='file_path', description='Selected file path')],
    },
    'copy2folder': {
        'name': 'Copy Files to Folder',
        'description': 'This node is used to copy multiple files connected to the input handle into same folder. Use the `folder_name` parameter to specify the name of the output folder.',
        'args': MoveFilesToFolderSchema,
        'input_handles': [Handle(name='files', description='Input files')],
        'output_handles': [Handle(name='folder_path', description='Folder path')],
    },
    'collect_mount_dirs': {
        'name': 'Collect Mount Directories',
        'description': 'This node collects multiple upstream tool working directories and outputs a mount-list manifest. The output is intended to be connected to a tool directory input that supports mountlist expansion.',
        'args': SimpleSchema,
        'input_handles': [Handle(name='dirs', description='Tool working directories')],
        'output_handles': [Handle(name='mount_dir_list', description='Mount-list manifest for downstream tool input')],
    },
    'compress': {
        'name': 'Compress',
        'description': 'Compresses one upstream file or directory. Set `compressed_file_name` to the desired output name; its suffix selects gzip, bzip2, xz, ZIP, or a TAR variant. Stream formats such as `.gz` support files only, while archive formats also support directories.',
        'args': CompressSchema,
        'input_handles': [Handle(name='input_path', description='File or directory to compress')],
        'output_handles': [Handle(name='compressed_path', description='Generated compressed file')],
    },
    'decompress': {
        'name': 'Decompress',
        'description': 'Automatically detects and decompresses gzip, bzip2, xz, ZIP, TAR, and compressed TAR inputs. The output path is a file for stream compression and a directory for archives.',
        'args': SimpleSchema,
        'input_handles': [Handle(name='compressed_path', description='Compressed file or archive')],
        'output_handles': [Handle(name='output_path', description='Decompressed file or extracted directory')],
    },
    'rename_file': {
        'name': 'Rename File',
        'description': 'This node is used to rename a file. Connect the input handle of this node to the output handle representing the file you want to rename. Use the `new_file_name` parameter to specify the new name for the file. The output handle of this node will provide the path to the renamed file.',
        'args': RenameFileSchema,
        'input_handles': [Handle(name='old_file', description='Old file to be renamed')],
        'output_handles': [Handle(name='renamed_file', description='Renamed file')],
    },
}

RESOURCES: dict[str, NodeSchema] = {
    'sequence': {
        'name': 'Pair-end Reads Input',
        'description': 'Loads one paired-end sequencing input as R1 and R2. Each path may be a plain filesystem path, `sample:<tag>`, or `proj:<keyword>`. Use the general `file` resource for single-end reads and other ordinary file or directory inputs.',
        'args': SequenceResourceSchema,
        'input_handles': [],
        'output_handles': [
            Handle(name='r1', description='r1 Raw sequence data'),
            Handle(name='r2', description='r2 Raw sequence data'),
        ],
    },
    'file': {
        'name': 'File / Directory Input',
        'description': 'Loads any existing file or directory from a plain filesystem path, a sample-level `sample:<tag>` reference, or a project-level `proj:<keyword>` reference. Use this general path resource for non-paired inputs such as single-end reads, custom annotations, images, result files, and directories. Prefer dedicated database and reference-genome resource nodes when applicable.',
        'args': FileInputSchema,
        'input_handles': [],
        'output_handles': [Handle(name='file_path', description='Resolved file or directory path')],
    },
    'text_file': {
        'name': 'Text File',
        'description': 'This node creates a UTF-8 text file from inline content. Use `file_name` for the output file name and `content` for its exact text content. It is intended for small configuration files, sample lists, and other text inputs required by downstream tools.',
        'args': TextFileResourceSchema,
        'input_handles': [],
        'output_handles': [Handle(name='file_path', description='Generated text file path')],
    },
    'db': {
        'name': 'Biological Database',
        'description': 'This node is used to load a specific biological database. Before adding it, use `resource_node_manager` with action="search_db" to find available databases. If no suitable database exists, use action="create_db" to create a downloadable database record, then pass the returned integer `db_id` parameter when adding this node.',
        'args': DbResourceSchema,
        'input_handles': [],
        'output_handles': [Handle(name='file_path', description='db file path')],
    },
    'ncbi_genome': {
        'name': 'NCBI Reference Genome',
        'description': 'This node is used to load a standard reference genome from NCBI RefSeq. It requires a preceding string variable node input to specify the species name or TaxID of the reference genome to be used.',
        'args': SimpleSchema,
        'input_handles': [
            Handle(
                name='species_name',
                description='(Optional) String variable, species name of the reference gene to be used',
            ),
            Handle(
                name='ncbi_tax_id',
                description='(Optional) String variable, String variable, NCBI taxonomy ID of the species of the reference gene to be used',
            ),
        ],
        'output_handles': [
            Handle(name='genome_fasta', description='Gene fasta file of the reference genome'),
            Handle(name='annotation_gff', description='GFF annotation file of the reference genome'),
            Handle(name='annotation_gtf', description='GTF annotation file of the reference genome'),
            Handle(name='bowtie2_index', description='Bowtie2 index of the reference genome'),
            Handle(name='bwa_index', description='BWA index of the reference genome'),
            Handle(name='hisat2_index', description='HISAT2 index of the reference genome'),
            Handle(name='minimap2_index', description='minimap2 index of the reference genome'),
            Handle(name='star_index', description='STAR index of the reference genome'),
        ],
    },
    'GRCh38': {
        'name': 'GRCh38',
        'description': 'This node loads the human reference genome sequence and annotation released by Gencode, based on the GRCh38 assembly.',
        'args': SimpleSchema,
        'input_handles': [],
        'output_handles': [
            Handle(name='genome_fasta', description='Gene fasta file of the reference genome'),
            Handle(name='annotation_gff', description='GFF annotation file of the reference genome'),
            Handle(name='annotation_gtf', description='GTF annotation file of the reference genome'),
            Handle(name='bowtie2_index', description='Bowtie2 index of the reference genome'),
            Handle(name='bwa_index', description='BWA index of the reference genome'),
            Handle(name='hisat2_index', description='HISAT2 index of the reference genome'),
            Handle(name='minimap2_index', description='minimap2 index of the reference genome'),
            Handle(name='star_index', description='STAR index of the reference genome'),
        ],
    },
    'GRCm39': {
        'name': 'GRCm39',
        'description': 'This node is used to loads Gencode mouse reference genome data based on the GRCm39 assembly',
        'args': SimpleSchema,
        'input_handles': [],
        'output_handles': [
            Handle(name='genome_fasta', description='Gene fasta file of the reference genome'),
            Handle(name='annotation_gff', description='GFF annotation file of the reference genome'),
            Handle(name='annotation_gtf', description='GTF annotation file of the reference genome'),
            Handle(name='bowtie2_index', description='Bowtie2 index of the reference genome'),
            Handle(name='bwa_index', description='BWA index of the reference genome'),
            Handle(name='hisat2_index', description='HISAT2 index of the reference genome'),
            Handle(name='minimap2_index', description='minimap2 index of the reference genome'),
            Handle(name='star_index', description='STAR index of the reference genome'),
        ],
    },
    'value_string': {
        'name': 'String Value',
        'description': 'This node encapsulates a string variable, which can be used as input for other nodes. Use the `value` parameter to specify the content of string',
        'args': ValueStringResourceSchema,
        'input_handles': [],
        'output_handles': [Handle(name='value', description='string value')],
    },
}

# Import edge check functions at the end to avoid circular imports
from llm.utils.edge_validation import check_ncbi_genome_edges, check_processor_edges  # noqa: E402

# Add edge_check to all processor nodes
for processor_name in PROCESSORS:  # noqa: PLC0206
    PROCESSORS[processor_name]['edge_check'] = check_processor_edges

# Add edge_check to ncbi_genome resource node
RESOURCES['ncbi_genome']['edge_check'] = check_ncbi_genome_edges
