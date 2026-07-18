---
name: CROSS_WORKFLOW_FILE_TRANSFER
description: Explains how to publish a workflow output with global_mark and load it in another workflow through the general file resource and a proj reference
---

# Skill: Cross-Workflow File Transfer

跨工作流传递单个文件或目录时，先使用处理器节点 `global_mark` 将上游输出注册为项目级路径，再在其他工作流中使用通用 `file` 资源节点和 `proj:<keyword>` 引用读取它。系统不存在单独的 `global_file` 资源节点。

## 节点说明

### global_mark 节点

处理器节点 `global_mark` 用于在某一个工作流中标记一个文件或目录，使其可以被同一项目中的其他工作流引用。将它的输入连接到需要发布的上游路径。
该节点有两个需要指明的参数：

- `mark_name`: 这是你为该文件指定的唯一标识符，其他工作流将通过这个名字来引用该文件。
- `description`: 简要说明该路径来自哪个工作流以及包含什么内容。

`mark_name` 只填写关键字本身，不要添加 `proj:` 前缀。

### file 资源节点

在消费端工作流中添加 `file` 资源节点，将其 `file_path` 设置为 `proj:<mark_name>`。任务运行时会从项目映射中解析出 `global_mark` 注册的实际路径。该路径既可以是文件，也可以是目录。

一个 `mark_name` 对应一个当前项目路径；这种机制不会自动收集多个样本的同名输出。如果需要多样本聚合，应使用专门的集合设计，而不是假设 `proj:` 会返回一个文件夹集合。

## 示例

在一个示例项目中，工作流 A 生成项目共享的参考索引目录，工作流 B 使用该目录进行比对。

1. 在工作流 A 中添加 `global_mark`，将索引目录标记为 `aligner_index`。
2. 在工作流 B 中添加 `file` 资源节点，设置 `file_path` 为 `proj:aligner_index`。
3. 将该节点的 `file_path` 输出连接到工具的索引目录输入。

## 辅助工具

设计工作流前，可以通过当前项目资源查看能力确认已有的 `proj:` 引用及其说明，避免虚构关键字。
