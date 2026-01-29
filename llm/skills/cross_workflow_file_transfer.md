---
name: CROSS_WORKFLOW_FILE_TRANSFER
description: aaa
---

# Skill: Cross-Workflow File Transfer

为了在不同工作流之间引用文件，你需要使用到两种节点，处理器节点`global_mark`，以及资源节点`global_file`。

## 节点说明

### global_mark 节点

处理器节点 `global_mark` 用于在某一个工作流中标记一个文件，使其可以被其他工作流引用。
使用方法就是将该节点的input_handles连接到代表对应文件的output_handles上。
该节点有两个需要指明的参数：

- `mark_name`: 这是你为该文件指定的唯一标识符，其他工作流将通过这个名字来引用该文件。
- `description`: 这是对该文件的简要描述，需要写明两点》这个文件来着哪个工作流，是一个是什么文件。

### global_file 节点

资源节点 `global_file` 用于在当前工作流中引用其他工作流中已经标记的文件。
使用方法是设置该节点的 `mark_name` 参数为你想要引用的文件的标识符。
这一操作会将该项目中所有运行了的工作流实例中对应标识符的文件都引用进来，并放置在一个文件夹中。此节点的输出就是一个包含所有这些文件的文件夹路径。

## 示例

在一个示例项目中，有工作流A和工作流B。A负责测序数据的指控和比对，B负责变异检测。

1. 在工作流A中，添加一个 `global_mark` 节点，将比对结果文件标记为 `aligned_bam`，并描述为“工作流A的比对结果BAM文件”。
2. 在工作流B中，添加一个 `global_file` 节点，设置其 `mark_name` 为 `aligned_bam`。
   那么，我针对16个样本运行的工作流A中生成的16个BAM文件就会被引用到工作流B中，，对应的 `global_file` 节点的输出将是一个包含这16个BAM文件的文件夹路径。

## 辅助工具

为了查看当前项目中已经使用的标记名称，你可以使用 `assist_tool` 工具，并传入参数 `action='view_global_marks'`，无需额外参数。