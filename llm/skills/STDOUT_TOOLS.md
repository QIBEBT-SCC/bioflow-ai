---
name: STDOUT_TOOLS
description: 当工具只能使用标准输出流作为输出时撰写配置文件的实例
---

# Skill: Tools with stdout as output

以工具`seqtk seq`为例，该工具使用标准输出流作为输出，实际命令如下所示：

```bash
seqtk seq -q 30 S1.fq.gz > /data/output/S1.fa 2> log.txt
```

## 文件挂载

```json
[
  {
    "name": "fq_file",
    "description": "FASTQ sequence file",
    "file_path": "{fq_file}",
    "mount_path": "/data/raw",
    "file_type": "INPUT",
    "is_report": false,
    "is_log": false
  },
  {
    "name": "fa_file",
    "description": "FASTA sequence file",
    "file_path": "{output_dir}/{sample_name}.fa",
    "mount_path": "/data/output",
    "file_type": "OUTPUT",
    "is_report": false,
    "is_log": false
  },
  {
    "name": "log",
    "description": "log",
    "file_path": "{output_dir}/00_run.log",
    "mount_path": "/data/output",
    "file_type": "OUTPUT",
    "is_report": false,
    "is_log": true
  }
]
```

## 参数设置

由于涉及到重定向输出，难以用简单的{dynamic_params}来实现参数模板的定义，所以这类工具不需要设置`dynamic_params`参数，而是直接在模板中使用文件挂载中定义的文件名作为变量

```json
{
  "dynamic_params": [],
  "immutable_static_params": "",
  "modifiable_static_params": "-q 30",
  "command_template": "seqtk seq {static_params} {fq_file} > {fa_file} 2> {log}"
}
```

## 核心要点

1. 此模式不需要再设置dynamic_params
2. 在此模式下，模板字符串中允许使用的变量有：
    - `file_mounts`中的`name`值（文件变量）
    - `{output_dir}`和`{sample_name}`，直接使用用于构建动态参数（输出路径、文件前缀等）
    - `{static_params}`
