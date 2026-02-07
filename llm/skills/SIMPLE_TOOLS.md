---
name: SIMPLE_TOOLS
description: 常规命令行工具配置文件撰写的实际示例，适用于大部分命令行工具
---

# Skill: Example Configuration for Standard Tools

以samtools sort为例，该工具的命令为：

```bash
samtools sort --write-index -@ 32 -o /data/output/A1.sorted.bam A1.bam > /data/output/11_run.log 2>&1
```

## 文件挂载

首先根据信息设置此工具的文件挂载：

```json
[
  {
    "name": "bam_in",
    "is_log": false,
    "file_path": "{bam_in}",
    "file_type": "INPUT",
    "is_report": false,
    "mount_path": "/data/raw",
    "description": "Input BAM/SAM file to be sorted by genomic coordinate"
  },
  {
    "name": "bam_sorted",
    "is_log": false,
    "file_path": "{output_dir}/{sample_name}.sorted.bam",
    "file_type": "OUTPUT",
    "is_report": false,
    "mount_path": "/data/output",
    "description": "Sorted BAM file organized by genomic coordinate"
  },
  {
    "name": "log",
    "is_log": true,
    "file_path": "{output_dir}/11_run.log",
    "file_type": "OUTPUT",
    "is_report": false,
    "mount_path": "/data/output",
    "description": "log"
  }
]
```

## 参数设置

此命令中包含了多种类型的参数

```json
{
  "dynamic_params": [
    {
      "index": 0,
      "command": "{bam_in}",
      "description": "Input BAM/SAM file to be sorted by genomic coordinate",
      "is_position": true
    },
    {
      "index": null,
      "command": "-o {bam_sorted}",
      "description": "Output sorted BAM file",
      "is_position": false
    }
  ],
  "immutable_static_params": "--write-index",
  "modifiable_static_params": "-@ 32",
  "command_template": "samtools sort {static_params} {dynamic_params} {position_params} > {log} 2>&1"
}
```

可以看到最终的command_template中包含了以下变量：

- `static_params`：由两种不同的静态参数自动拼接
- `dynamic_params`和`position_params`：根据is_position从原本的dynamic_params中拆分出来
- `log`：日志文件，直接使用之前文件挂载中的文件名

## 核心要点

- 只要参数涉及到输出路径（`{output_dir}` 变量）和样本名称（`{sample_name}`变量），则应当放在dynamic_params中
- 类似线程数、物种名称这样的参数，与输出路径和文件名无关，放在对应的静态参数中
- command_template中的dynamic_params变量不等于参数设置中的dynamic_params，不要混淆