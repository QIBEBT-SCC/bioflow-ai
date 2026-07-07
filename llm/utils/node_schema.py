from collections.abc import Callable
from typing import NotRequired, TypedDict

from pydantic import BaseModel

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


PROCESSORS: dict[str, NodeSchema] = {
    'global_mark': {
        'name': 'Global Mark',
        'description': 'This node is used to mark a file as a project file. By connecting the input handle of this node to the output handle representing the corresponding file. Need two parameters: `mark_name`, which is a unique identifier for the file that other workflows will use to reference it, and `description`, which provides a brief explanation of the file',
        'args': GlobalMarkSchema,
        'input_handles': [Handle(name='file', description='Files to be marked for cross-workflow use')],
        'output_handles': [],
    },
    'bind_param': {
        'name': 'Bind Parameter',
        'description': '这个节点的作用是将一个来自其他节点的文件绑定上指定的参数名称供 CODE 节点使用。输出只能够作为code节点的输入。',
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
    'gzip': {
        'name': 'Gzip',
        'description': 'This node is used to compress the input file using Gzip.',
        'args': SimpleSchema,
        'input_handles': [Handle(name='file', description='Origin file')],
        'output_handles': [Handle(name='zipped_file', description='Gzipped file')],
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
        'description': 'This node is used for input of common paired-end biological sequencing files. For non-paired-end data, use file node instead.',
        'args': SequenceResourceSchema,
        'input_handles': [],
        'output_handles': [
            Handle(name='r1', description='r1 Raw sequence data'),
            Handle(name='r2', description='r2 Raw sequence data'),
        ],
    },
    'file': {
        'name': 'File Input',
        'description': 'This node is used for input of general files. A note node can be added as appropriate to describe what the input file is.',
        'args': FileInputSchema,
        'input_handles': [],
        'output_handles': [Handle(name='file_path', description='file path')],
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
