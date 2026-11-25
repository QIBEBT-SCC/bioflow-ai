---
name: sample-registration
description: '当 /work/SAMPLES.md 不存在、需要检查原始数据、划分并注册样本与项目共享文件以及编写 SAMPLES.md 时使用。'
---

# 样本整理与注册指南

当 `/work/SAMPLES.md` 不存在时，按以下流程和规范完成数据检查、样本与共享文件注册，并编写 `/work/SAMPLES.md`：

1. 结合用户对数据和需求的描述，在 `/app/data` 下用 `run_command` 检查目录结构并抽查代表性文件。
2. 按下文规则划分样本后调用 `add_samples`；元数据相同的样本在同一次调用中批量创建。
3. 用 `add_sample_file` 为每个样本注册实际输入（文件或目录）。
4. 用 `add_project_file` 注册参考基因组、注释、索引/数据库目录等项目共享资源（如有）。
5. 如需核对当前项目数据库中已登记的样本和共享文件，可按需调用 `list_project_samples`。
6. 按下文格式用 `apply_patch` 在 `/work` 写入 `SAMPLES.md`。

## 样本划分

系统中的一个样本，是一次样本级工作流运行所需的完整输入集合，不一定等于一个文库、一个文件名前缀或一个子目录。

- 每个都能独立跑完整流程的重复（如 `A_R1/R2`、`B_R1/R2`）→ 两个样本，各含 `r1`、`r2`。
- 必须一起交给下游命令的配对数据（处理/对照、肿瘤/正常、前/后等）→ 同一个复合样本，文件用角色前缀标签区分，如 `treatment_r1`、`control_r1`，并在元数据中记录各自角色。
- 所有独立样本共用的文件（参考基因组、注释、BED 等）→ 项目级。
- 对照不会因为名为“对照”就成为项目级文件；只有多个处理样本有意共用同一个混合对照时，才可登记为 `proj:pooled_control_r1` 等。
- 只登记下游工作流会直接读取的资源，用户提供的原始文件不一定都要注册。
  给人看的说明性表格，系统中没有对应的处理流程，注册后下游也用不上：应从中提取物种、条形码、分组等信息，写入对应样本的元数据和 `SAMPLES.md`，而不是注册文件本身。

### 按逻辑资源注册（文件 vs 目录）
`add_sample_file` 和 `add_project_file` 既可注册单个文件，也可注册整个目录；**注册单位是下游工具消费的逻辑资源，而不是把目录里的每一个文件逐个登记**。
- **多文件索引或数据库 → 整体注册为一个目录**：如预构建的 Bowtie2（`*.bt2`）、BWA、HISAT2（`*.ht2`）、STAR、Salmon 索引目录，或 Kraken2、BLAST、MetaPhlAn 等本地数据库目录，只需用 `add_project_file` 将整个目录注册为**一条** `proj:<keyword>`，而不是把其中的每个索引分片或数据库小文件逐一注册。在 `description` 和 `SAMPLES.md` 中写明该目录的用途、索引前缀（如前缀 `hg38`）及包含的主要文件类型即可。
- **独立的不同角色文件 → 分别注册**：若同一目录下混放着彼此独立、下游分别引用的参考文件（如 `genome.fa`、`genes.gtf`、`targets.bed`），仍应按各自语义分别注册为独立的 `proj:<keyword>`，不要把无关文件打包成一个目录。
- 同理，若单个样本的输入本身就是需按目录交给下游的目录型数据，直接注册该目录为一个 `sample:<tag>`，不要拆散其中的零散文件。

## 样本名称、元数据与项目文件描述

### 样本名称
取自实际文件名或子目录名，不要随意编造；复合样本使用体现配对关系的简洁名称，如 `SPT5_T0_with_INPUT`。名称在项目内必须唯一。

### 样本元数据
通过 `add_samples` 写入，描述输入数据本身及其来源，与之后要做的分析无关。常用可选键：

| 键 | 含义 |
|---|---|
| `organism` | 物种学名，如 `Homo sapiens` |
| `assay` | 产生数据的实验类型，如 `rna-seq`、`wgs`、`chip-seq`，不是计划中的分析 |
| `layout` | `single-end` 或 `paired-end`，需经确认 |
| `platform` | 测序或采集平台，如 `illumina`、`oxford_nanopore`、`pacbio`，不是文件格式 |

- 依据文件内容、头信息、配对情况、样本表和用户说明确定。
- 不确定或不适用字段直接省略，不要填猜测值或 `unknown`；其他已确认的信息可用自定义键保存。
- 复合样本中，只有所有组成文件都适用的值才写入通用键，不同的属性用带角色前缀的自定义键保存。

### 项目文件描述
用于 `add_project_file` 的 `description` 与 `SAMPLES.md` 中的说明：

- 除生物学或分析角色外，还要写出已确认、且可能影响下游工具能否直接使用的特征，如压缩或打包格式、与扩展名不一致的实际格式、表头/分隔符/列数、异常或截断记录、标识符和序列名的取值范围、相关文件之间标识符是否一致等。
- 只检查与该文件下游用途相关的项目；不要把每个文本文件都当作表格，也不要仅凭后缀套用格式规则。
- 只客观陈述事实，例如“GTF 为 gzip 压缩”“注释文件存在 9 列和 11 列两种行”；是否需要预处理放到计划中决定。
- 大文件用流式命令给出汇总统计和异常情况，不要复制原始内容。

## Tag 和 Keyword 命名

常见场景的 `tag` 命名参考：

| 场景 | tag |
|---|---|
| 双端测序 R1 / R2 | `r1` / `r2` |
| 单端测序 | `read` |
| 质谱 | `spectrum` |
| 图像 | `img` |
| 其他 | `contig`、`bam` 等 |

- 只使用小写字母和下划线，不超过 50 个字符，且需具备语义。
- 一个样本包含多组文件时加描述性前缀，如 `tumor_r1`、`normal_r1`、`light_img`。
- `tag` 在样本内唯一，`keyword` 在项目内全局唯一。

## SAMPLES.md 格式

`SAMPLES.md` 的撰写示例如下所示，正文与标题遵循当前用户语言设置：

```markdown
# Samples Info

### Sample metadata
- Sample count: N
- [Confirmed `organism`, `assay`, `layout`, `platform`, and custom metadata, grouped by samples when values differ]

### Sample-level references
[Describe which samples or composite sample contain these tags:]
- `sample:xxx` — [description]
- `sample:xxx` — [description]
[List custom tags if any]

### Project-level references
- `proj:xxx` — [description]
[Omit this section if none]

### Notes
[Special groupings, mixed data types, or anything downstream agents should be aware of]
```

不要列出具体文件路径，只描述下游可以通过 `sample:` 和 `proj:` 引用的内容。

## 数据检查技巧

- 同一批角色和结构相同的大量文件，抽查代表性文件即可，再统一应用已确认的特征。
- 对执行关键的单个参考或元数据文件，如果一次流式命令就能扫描全文，优先做全量统计，避免只看开头而漏掉少量异常记录。
