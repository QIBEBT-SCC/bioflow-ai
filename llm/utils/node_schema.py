from typing import NotRequired, Type, TypedDict

from pydantic import BaseModel

from app.schemas.tools import Handle


class NodeSchema(TypedDict):
    name: str
    description: NotRequired[str]
    args: Type[BaseModel]
    input_handles: list[Handle]
    output_handles: list[Handle]


class GlobalMarkSchema(BaseModel):
    mark_name: str
    description: str


class SelectFileInFolderSchema(BaseModel):
    file_name: str


class RenameFileSchema(BaseModel):
    new_file_name: str


class SequenceResourceSchema(BaseModel):
    pass


class FileResourceSchema(BaseModel):
    pass


class GlobalFileSchema(BaseModel):
    mark_name: str


class DbResourceSchema(BaseModel):
    db_id: int


class GenomeResourceSchema(BaseModel):
    pass


class ValueStringResourceSchema(BaseModel):
    pass


PROCESSORS: dict[str, NodeSchema] = {
    'global_mark': {
        'name': 'Global Mark',
        'description': 'This node is used to mark a file for cross-workflow usage. By connecting the input handle of this node to the output handle representing the corresponding file. Need two parameters: `mark_name`, which is a unique identifier for the file that other workflows will use to reference it, and `description`, which provides a brief explanation of the file, including which workflow it comes from and what type of file it is.',
        'args': GlobalMarkSchema,
        'input_handles': [Handle(name='file', description='Files to be marked for cross-workflow use')],
        'output_handles': [],
    },
    'select_file_in_folder': {
        'name': 'Select File in Folder',
        'description': 'This node is used to select a specific file from a folder. Connect the input handle of this node to the output handle representing the folder. Use the `file_name` parameter to specify the name of the file you want to select within the folder. The output handle of this node will provide the path to the selected file.',
        'args': SelectFileInFolderSchema,
        'input_handles': [Handle(name='folder_path', description='Folder path')],
        'output_handles': [Handle(name='file_path', description='Selected file path')],
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
        'description': 'This node is used for input of common paired-end biological sequencing files. No additional parameters are required.',
        'args': SequenceResourceSchema,
        'input_handles': [],
        'output_handles': [
            Handle(name='r1', description='r1 Raw sequence data'),
            Handle(name='r2', description='r2 Raw sequence data'),
        ],
    },
    'file': {
        'name': 'File Input',
        'description': 'This node is used for input of general files. No additional parameters are required. A note node can be added as appropriate to describe what the input file is.',
        'args': FileResourceSchema,
        'input_handles': [],
        'output_handles': [Handle(name='file_path', description='file path')],
    },
    'global_file': {
        'name': 'Global File Input',
        'description': 'This node is used to reference files marked in other workflows. Set the `mark_name` parameter of this node to the identifier of the file you want to reference. This operation will bring in all files with the corresponding identifier from all executed workflow instances in the project and place them in a folder. The output of this node is the folder path containing all these files.',
        'args': GlobalFileSchema,
        'input_handles': [],
        'output_handles': [Handle(name='file_path', description='file path')],
    },
    'db': {
        'name': 'Biological Database',
        'description': 'This node is used to load a specific biological database. Before use, use the `search_biodb` tool to search for available databases. If data is available, pass an integer `db_id` parameter to indicate your chosen database. If no parameters are found, call the `ask_human` parameter to request staff to add them.',
        'args': DbResourceSchema,
        'input_handles': [],
        'output_handles': [Handle(name='file_path', description='db file path')],
    },
    'genome': {
        'name': 'NCBI Reference Genome',
        'description': 'This node is used to load a standard reference genome from NCBI. It requires a preceding string variable node input to specify the species name or TaxID of the reference genome to be used. Meanwhile, this node provides not only the reference genome and annotation but also pre-built indexes for commonly used alignment tools; in most cases, you can directly use the index files provided by this node. Note that if the reference genome is not from an NCBI source, you need to use the file node to load the reference genome file and manually build the index, and add a note node for explanation.',
        'args': GenomeResourceSchema,
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
    'value_string': {
        'name': 'String Value',
        'description': 'This node encapsulates a string variable, which can be used as input for other nodes.',
        'args': ValueStringResourceSchema,
        'input_handles': [],
        'output_handles': [Handle(name='value', description='string value')],
    },
}
