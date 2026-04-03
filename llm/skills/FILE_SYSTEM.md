---
name: FILE_SYSTEM
description: How to use the file auto-loading mechanism of this system
---

# 文件系统自动加载机制

本系统支持在工作流节点中使用**路径引用**来动态加载文件，而不必硬编码具体路径。引用在任务执行时解析，分为两个层级。

## 引用语法

| 格式               | 层级  | 说明          |
|------------------|-----|-------------|
| `proj:<keyword>` | 项目级 | 引用项目的共享文件   |
| `sample:<tag>`   | 样本级 | 引用当前样本的某个文件 |
| 普通路径字符串          | —   | 直接使用，不做解析   |

## 项目级引用：`proj:<keyword>`

项目可以维护一张关键字→路径的映射表（`project_file_mappings`），用于存放跨样本共享的资源，例如参考基因组索引、数据库路径等。

**适用场景**：所有样本共用同一份资源（如比对索引、注释文件）

## 样本级引用：`sample:<tag>`

每个样本可以有多个文件，每个文件有一个 `tag` 标识。 系统默认的 tag 分配规则：

| data_type      | 默认 tag     |
|----------------|------------|
| SEQUENCING_R1  | `r1`       |
| SEQUENCING_R2  | `r2`       |
| SEQUENCING（单端） | `single`   |
| SPECTRUM       | `spectrum` |
| IMAGE          | `image`    |

**适用场景**：每个样本内各自的不同文件

## 节点中的使用方式

在系统中，可以使用以下节点读取(写入)动态文件类型：

- resource_sequence: 读取样本级文件
- resource_file: 读取项目级、样本级文件
- global_mark(处理器节点): 将一个文件标为项目级引用

## 典型场景示例

### 场景一：单样本标准测序分析

使用默认的 tag 名称加载样本中的双端序列：

```
{
  "r1": "sample:r1",
  "r2": "sample:r2"
}
```

### 场景二：一个样本包含阳性 + 对照两组测序数据

上传文件时指定自定义 tag：

- 阳性 R1 → tag: `r1_pos`，阳性 R2 → tag: `r2_pos`
- 对照 R1 → tag: `r1_ctrl`，对照 R2 → tag: `r2_ctrl`

工作流中使用两个 resource_sequence 节点：

```
// 节点1（阳性）
{
  "r1": "sample:r1_pos",
  "r2": "sample:r2_pos"
}

// 节点2（对照）
{
  "r1": "sample:r1_ctrl",
  "r2": "sample:r2_ctrl"
}
```

### 场景三：共享bed注释文件

首先在全局工作流中使用相应的工具得到了一个bed文件，将这个文件连接至 global_mark 节点，标记为全局文件：

```
{
  "mark_name": "exon_bed",
  "description": "..."
}
```

注意这里 `mark_name` 字段直接输入 keyword 就行，不用在前面加上 proj:

在其它工作流中使用 resource_file 节点引用：

```json
{
  "file_path": "proj:exon_bed"
}
```

## 注意事项

- keyword 全项目唯一
- 同一样本内 tag 唯一
- tag 和 keyword 命名使用小写字母+下划线的形式, 不要超过50个字符
- tag 和 keyword 的名字由你来决定，用户使用时根据你的定义配置相应的文件
