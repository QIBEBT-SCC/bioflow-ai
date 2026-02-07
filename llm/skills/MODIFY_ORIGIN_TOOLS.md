---
name: MODIFY_ORIGIN_TOOLS
description: 当工具直接修改原始文件而不产生新的输出时如何编写配置文件
---

# Example Configuration for Standard Tools

例如，工具 `cooler balance`会直接修改原始文件，该工具的命令形如这样：

```
cooler balance --force --nproc 32 H1.cool 2>&1 /data/output/11_run.log
```

## 文件挂载

原本由于工具直接修改源文件，所以输入和输出都是同一个文件。但是为了保证文件独立性，我们需要将文件复制到独立的输出路径下再进行修改。
因此，需要分别准备原始文件（输入）和复制后的文件（输出）

```json
[
  {
    "name": "cool_path",
    "is_log": false,
    "file_path": "{cool_path}",
    "file_type": "INPUT",
    "is_report": false,
    "mount_path": "/data/output",
    "description": "Input .cool Hi-C contact matrix (Cooler format)"
  },
  {
    "name": "balanced_cool_path",
    "is_log": true,
    "file_path": "{output_dir}/{sample_name}.cool",
    "file_type": "OUTPUT",
    "is_report": false,
    "mount_path": "",
    "description": "Balanced .cool Hi-C contact matrix (Cooler format)"
  },
  {
    "name": "log",
    "is_log": true,
    "file_path": "{output_dir}/11_run.log",
    "file_type": "OUTPUT",
    "is_report": false,
    "mount_path": "/data/output",
    "description": "Execution log capturing stdout and stderr from cooler balance."
  }
]
```

### 命令模板

由于现在命令组成不再是单一命令，情况复杂，所以允许不设置动态参数，直接在命令模板中使用文件变量：

```json
{
  "dynamic_params": [],
  "immutable_static_params": "--force",
  "modifiable_static_params": "--nproc 32",
  "command_template": "cp {cool_path} {balanced_cool_path} && cooler balance {static_params} {balanced_cool_path} 2>&1 {log}"
}
```

## 核心要点

1. 输入文件在挂载中对应：输入类型的源文件和位于输出路径下的实际操作文件（同时也是输出文件）
2. 命令需要先将源文件复制为输出文件，然后命令在输出文件上跑
3. 在此模式下，模板字符串中允许使用的变量有：
    - `file_mounts`中的`name`值（文件变量）
    - `{output_dir}`和`{sample_name}`，直接使用用于构建动态参数（输出路径、文件前缀等）
    - `{static_params}`