import re
from collections.abc import Callable
from pathlib import Path
from typing import NotRequired, TypedDict

from pydantic import BaseModel, Field, field_validator, model_validator

from app.schemas.archive import CompressionFormat, resolve_compression_format
from app.schemas.tools import Handle

# Forward declaration for type hint
from llm.states import BuildEdge

MARK_NAME_MAX_LENGTH = 50


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
class ArtifactMarkSchema(BaseModel):
    mark_name: str = Field(min_length=1, max_length=50, pattern=r'^[a-z][a-z0-9_]*$')
    description: str


class SampleMarkCollectionSchema(BaseModel):
    mark_names: list[str] = Field(min_length=1)
    require_all_samples: bool = True

    @field_validator('mark_names')
    @classmethod
    def validate_mark_names(cls, mark_names: list[str]) -> list[str]:
        """Normalize duplicate mark names and reject invalid names."""
        pattern = re.compile(r'^[a-z][a-z0-9_]*$')
        normalized = list(dict.fromkeys(mark_names))
        invalid = [name for name in normalized if len(name) > MARK_NAME_MAX_LENGTH or pattern.fullmatch(name) is None]
        if invalid:
            msg = f'invalid mark names: {invalid}'
            raise ValueError(msg)
        return normalized


class SelectFileInFolderSchema(BaseModel):
    file_name: str


class JoinPathSchema(BaseModel):
    relative_path: str

    @field_validator('relative_path')
    @classmethod
    def validate_relative_path(cls, relative_path: str) -> str:
        """Require a non-empty relative path that cannot escape its base path."""
        path = Path(relative_path)
        if (
            not relative_path
            or relative_path == '.'
            or path.is_absolute()
            or '..' in path.parts
            or '\\' in relative_path
        ):
            msg = 'relative_path must be a non-empty relative path without parent traversal'
            raise ValueError(msg)
        return relative_path


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
    compression_format: CompressionFormat = Field(
        description=(
            'Compression format. Archive formats work with files and directories; '
            'stream formats `.gz`, `.bz2`, and `.xz` only work with regular files.'
        )
    )

    @field_validator('compressed_file_name')
    @classmethod
    def validate_compressed_file_name(cls, compressed_file_name: str) -> str:
        """Validate that the output is a plain file name."""
        if (
            not compressed_file_name
            or compressed_file_name in {'.', '..'}
            or Path(compressed_file_name).name != compressed_file_name
            or '\\' in compressed_file_name
        ):
            msg = 'compressed_file_name must be a single, non-empty file name'
            raise ValueError(msg)

        return compressed_file_name

    @model_validator(mode='after')
    def validate_compression_format(self) -> 'CompressSchema':
        """Require the selected format to match the output file suffix."""
        resolve_compression_format(self.compressed_file_name, self.compression_format)
        return self


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
    'project_mark': {
        'name': 'Project Mark',
        'description': 'Registers one project-level workflow output immediately when this node succeeds. Use only in project-level workflows. Later workflows load it with `proj:<mark_name>`. The mapping is stored with normal project files but tagged as dynamic, and is cleared when its producer workflow reruns.',
        'args': ArtifactMarkSchema,
        'input_handles': [Handle(name='file', description='Project-level file or directory to register')],
        'output_handles': [],
    },
    'sample_mark': {
        'name': 'Sample Mark',
        'description': 'Registers one output for the current sample immediately when this node succeeds. Use only in sample-level workflows. A later workflow for the same sample loads it with `sample:<mark_name>`; project-level aggregation uses the `sample_mark_collection` resource.',
        'args': ArtifactMarkSchema,
        'input_handles': [Handle(name='file', description='Current sample file or directory to register')],
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
    'join_path': {
        'name': 'Join Path',
        'description': 'Appends `relative_path` to an upstream file or directory path and returns the joined path. This node only constructs the path; it does not require the resulting path to exist.',
        'args': JoinPathSchema,
        'input_handles': [Handle(name='base_path', description='Base file or directory path')],
        'output_handles': [Handle(name='path', description='Joined path')],
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
        'description': 'Compresses one upstream file or directory. Set `compression_format` explicitly and use the matching suffix in `compressed_file_name`. Supported formats are `.zip`, `.tar`, `.tar.gz`, `.tar.bz2`, `.tar.xz`, `.gz`, `.bz2`, and `.xz`. Stream formats `.gz`, `.bz2`, and `.xz` support regular files only; archive formats also support directories.',
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
        'description': 'Loads one paired-end sequencing input as R1 and R2. Each path may be a plain filesystem path or a `sample:<tag>`/`proj:<keyword>` reference. Those references may point to initial or dynamically marked files. Use the general `file` resource for single-end reads and other ordinary files or directories.',
        'args': SequenceResourceSchema,
        'input_handles': [],
        'output_handles': [
            Handle(name='r1', description='r1 Raw sequence data'),
            Handle(name='r2', description='r2 Raw sequence data'),
        ],
    },
    'file': {
        'name': 'File / Directory Input',
        'description': 'Loads any existing file or directory from a plain path or a `sample:<tag>`/`proj:<keyword>` mapping. Mappings may be initial inputs or dynamically registered workflow outputs. Use this general path resource for non-paired inputs such as single-end reads, custom annotations, images, result files, and directories.',
        'args': FileInputSchema,
        'input_handles': [],
        'output_handles': [Handle(name='file_path', description='Resolved file or directory path')],
    },
    'sample_mark_collection': {
        'name': 'Sample Mark Collection',
        'description': 'Collects files saved by `sample_mark` across all samples in a project. It returns a JSON file that lists the collected files and a read-only folder input for Docker tools. If the next tool needs both a generated manifest and the collected files, use the `adapt-sample-mark-collections` skill.',
        'args': SampleMarkCollectionSchema,
        'input_handles': [],
        'output_handles': [
            Handle(
                name='mount_collection',
                description=(
                    'Special read-only folder containing the collected files. Each file is located at the relative '
                    'path stored in its JSON entry `mount_name`. Connect this output directly to a Docker tool folder '
                    'input; do not connect it to CODE or file processors'
                ),
            ),
            Handle(
                name='collection_manifest',
                description=(
                    'JSON file listing the collected items. Each entry includes `sample_name`, `mark_name`, '
                    '`mount_name`, `source_path`, and `kind`. Python CODE should use `mount_name` when writing paths '
                    'for the next tool and must not try to open or copy `source_path`'
                ),
            ),
        ],
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
from llm.workflow_builder.validation import check_processor_edges, check_ncbi_genome_edges  # noqa: I001, E402

# Add edge_check to all processor nodes
for processor_name in PROCESSORS:  # noqa: PLC0206
    PROCESSORS[processor_name]['edge_check'] = check_processor_edges

# Add edge_check to ncbi_genome resource node
RESOURCES['ncbi_genome']['edge_check'] = check_ncbi_genome_edges
