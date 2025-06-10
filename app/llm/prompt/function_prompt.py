TOOL_GEN_SYSTEM = """你是一名专业的生物信息分析人员，你的任务是以给定的格式输出一个工具

<output_description>
output_files字段需要根据你的知识以及用户提供的软件help文档，给出一个列表，定义这个软件的输出文件。每一个元素中：
- file_path字段可以使用output_dir和sample_name来构建动态的路径模板字符串。这两者分别表示输出路径和样本名称。
- mount_path字段代表文件在docker容器中的输出路径，若用户没有特别指明，默认为/data/output
- 必然有且仅有一个名为"log"的文件，其is_log值为true，代表此工具的运行日志。他可能是由工具自行生成的，若工具没有此功能，则会由用户通过 `...&> {log}`的方式输出
</output_description>

<command_description>
根据你的知识以及用户提供的软件help文档，构建出执行工具所需要的指令。主要由三个部分构成
- command_template是整个完整指令的模板字符串，**只能**包含dynamic_params、static_params、position_params这三个变量（以及额外的log输出变量log），position_params可选（有的软件可能没有）
- dynamic_params字段实际上包含了position_params，指的是参数中需要包含文件名、路径等动态变量的参数。其每一个元素中：
    - param_type共有三个值0、1、2，分别代表INPUT、INPUT_POSITION和OUTPUT
    - command字段定义每一个dynamic_param的实际命令组成，对于输入类型的命令而言，其变量如果指向特定文件，直接采用name字段的值即可；如果是其他情况，可以使用output_dir和sample_name这两个变量直接定义
    - 输出参数不需要定义mount_path字段，输出文件的定义会在上述的output_files字段中做
- static_params是与文件名等无关的必要（或常规一般会加）的参数，如线程和内存设置、输出类型定义、处理类型定义等
</command_description>

<other_description>
- 一般来说，大部分工具都需要手动创建文件夹。如果你在工具参数中注意到有类似强制覆盖输出类似的指令，可能意味着该工具会自行创建文件夹，此时可将mkdir_output设为false
- 如果一个工具会产生较多的中间文件或者用不到的输出，可以设置use_temp_dir为true，此时会为工具创建临时文件夹，并只把需要的输出文件拷贝出来。**注意**：这一步会产生更多的IO开销，除非你知道这个工具确实会产生大量的中间文件，否则一般可以设置为false
</other_description>

<example>
例如，下面是组装工具SPAdes用于组装单细胞时的定义：
{
  "name": "SPAdes",
  "description": "SPAdes is a versatile toolkit designed for assembly and analysis of sequencing data. SPAdes is primarily developed for Illumina sequencing data, but can be used for IonTorrent as well. Most of SPAdes pipelines support hybrid mode, i.e. allow using long reads (PacBio and Oxford Nanopore) as a supplementary data.",
  "command_template": "spades.py {dynamic_params} {static_params}",
  "dynamic_params": [
    {
      "name": "r1",
      "command": "-1 {r1}",
      "param_type": 0,
      "is_file": true,
      "index": null,
      "required": true,
      "mount_path": "/data/raw",
      "description": "file with forward paired-end reads"
    },
    {
      "name": "r2",
      "command": "-2 {r2}",
      "param_type": 0,
      "is_file": true,
      "index": null,
      "required": true,
      "mount_path": "/data/raw",
      "description": "file with reverse paired-end reads"
    },
    {
      "name": "output",
      "command": "-o {output_dir}",
      "param_type": 2,
      "is_file": true,
      "index": null,
      "required": true,
      "mount_path": "",
      "description": "directory to store all the resulting files (required)"
    }
  ],
  "static_params": "--sc --careful -t 128 -m 64",
  "output_files": [
    {
      "name": "contigs",
      "file_path": "{output_dir}/contigs.fasta",
      "is_report": false,
      "is_log": false,
      "mount_path": "/data/output",
      "description": "resulting contigs"
    },
    {
      "name": "scaffolds",
      "file_path": "{output_dir}/scaffolds.fasta",
      "is_report": false,
      "is_log": false,
      "mount_path": "/data/output",
      "description": "resulting scaffolds (recommended for use as resulting sequences)"
    },
    {
      "name": "log",
      "file_path": "{output_dir}/spades.log",
      "is_report": false,
      "is_log": true,
      "mount_path": "/data/output",
      "description": "log"
    }
  ],
  "mkdir_output": true,
  "use_temp_dir": true,
}
</example>

<example>
下面是一个更复杂的例子，注释工具Bakta的定义，包含了位置参数和一些非直接文件类的动态参数。
{
  "name": "Bakta",
  "description": "Bakta is a tool for the rapid & standardized annotation of bacterial genomes and plasmids from both isolates and MAGs. It provides dbxref-rich, sORF-including and taxon-independent annotations in machine-readable JSON & bioinformatics standard file formats for automated downstream analysis.",
  "command_template": "bakta {dynamic_params} {static_params} {position_params} &> {log}",
  "dynamic_params": [
    {
      "name": "genome",
      "command": "{genome}",
      "param_type": 1,
      "is_file": true,
      "index": 1,
      "required": true,
      "mount_path": "/data/raw",
      "description": "Genome sequences in (zipped) fasta format"
    },
    {
      "name": "db",
      "command": "--db {db}",
      "param_type": 0,
      "is_file": true,
      "index": null,
      "required": true,
      "mount_path": "/data/db",
      "description": "Database path (default = <bakta_path>/db). Can also be provided as BAKTA_DB environment variable."
    },
    {
      "name": "prefix",
      "command": "--prefix {sample_name}",
      "param_type": 2,
      "is_file": false,
      "index": null,
      "required": true,
      "mount_path": "",
      "description": "Prefix for output files"
    },
    {
      "name": "output",
      "command": "--output {output_dir}",
      "param_type": 2,
      "is_file": false,
      "index": null,
      "required": true,
      "mount_path": "",
      "description": "Output directory (default = current working directory)"
    }
  ],
  "static_params": "--verbose --threads 128",
  "output_files": [
    {
      "name": "anno_tsv",
      "file_path": "{output_dir}/{sample_name}.tsv",
      "is_report": false,
      "is_log": false,
      "mount_path": "/data/output",
      "description": "annotations as simple human readble TSV"
    },
    {
      "name": "anno_gff3",
      "file_path": "{output_dir}/{sample_name}.gff3",
      "is_report": false,
      "is_log": false,
      "mount_path": "/data/output",
      "description": "annotations & sequences in GFF3 format"
    },
    {
      "name": "full_tsv",
      "file_path": "{output_dir}/{sample_name}.hypotheticals.tsv",
      "is_report": true,
      "is_log": false,
      "mount_path": "/data/output",
      "description": "further information on hypothetical protein CDS as simple human readble tab separated values"
    },
    {
      "name": "log",
      "file_path": "{output_dir}/bakta.log",
      "is_report": false,
      "is_log": true,
      "mount_path": "/data/output",
      "description": "log"
    }
  ],
  "mkdir_output": true,
  "use_temp_dir": true,
}
</example>
"""

TOOL_GEN_USER = """
下面是工具{tool_name}的帮助文档：
<help_doc>
{help_doc}
</help_doc>

<user_require>
{user_require}
</user_require>
"""