---
name: write-tool-memory
description: Write reusable shared Tool Memory entries when troubleshooting, diagnosis, or when the user asks to save a tool memory. Load this skill before calling write_node_memory.
---

# 编写工具 Memory

将排查、诊断或讨论中沉淀出的通用工具使用知识，写成可供所有用户复用的共享 Tool Memory，并提交给用户逐条确认后保存。

## 定位与读者

Tool Memory 的主要读者是**未来在全新工作流中构建、配置该 Tool 节点的智能体**（以及后续排错的助手）。因此每条 Memory 都应写成**前瞻性的通用使用指南**，而不是针对单次运行的故障复盘或修改记录。

## 适用范围与去重

1. `write_node_memory` 仅支持当前运行中的 Tool 节点，一次提交 1 到 3 条候选。
2. 定位目标节点时，使用 `workflow_reader` 中的本地 `node_id`；若节点位于普通子图或 ForEach 批量子图中，同时传入对应的 `subgraph_path` 与 `child_run_uid`。
3. 提交前先用 `workflow_reader(action="node", ...)` 查看目标节点已有的 `memories` 摘要；若已有覆盖相同主题的条目，避免重复写入。

## 编写规范

### 1. `description`（单行，最多 120 字）

概括这条 Memory 涵盖的**配置主题或使用场景**，便于后续建流智能体仅凭列表即可判断是否需要读取正文。

- 推荐写法：聚焦工具能力、参数语义或输入前提，例如：
  `Guidelines for configuring thread count (-@) and per-thread sort memory (-m) in samtools sort.`
- 避免写法：不要写成只在报错后才适用的条件句（如“当某次运行发生 OOM 时使用”）。

### 2. `content`（Markdown 正文）

建议按以下结构组织：

1. **原理与底层机制**：先解释工具参数、资源行为或输入输出约定背后的底层逻辑与常见陷阱（例如参数作用域、隐式内存放大、文件格式/标签依赖等）。
2. **面向新工作流的配置与使用建议**：给出具体、可落地的配置策略或上下游衔接方式（例如在 worker 资源配额未知或受限时，建议采用较保守的线程与内存参数起步；或在下游工具之前由上游节点提前准备好特定标签/索引）。
3. **权衡与适用前提（按需）**：简要说明调整参数带来的副作用、前提条件或边界（例如调低排序内存会增加临时文件读写与暂存空间占用）。

### 3. 禁止写入的内容

- 不写本次运行用过什么具体参数、用户反馈了什么、或修复中改动了哪些节点配置。
- 不写“尚未重跑验证 / Validation limits”等仅属于当前运行或单次修复的状态说明。
- 不编造未经实测的性能数据、加速比或绝对硬件需求。
- Memory 对所有用户共享，严禁包含项目私有路径、样本名、运行 UID 或敏感日志片段。

### 4. 示例对照

**推荐结构（以 `samtools sort` 线程与内存配置为例）**：

```markdown
### How memory and threads interact
In `samtools sort`, `-m` specifies the approximate memory limit **per sorting thread**, not a process-wide cap. Total sort buffer demand scales roughly with thread count (`-@`) multiplied by `-m`, plus extra overhead for BAM/CRAM decompression, compression, and runtime buffers. Aggressive `-@` and `-m` values can exceed container memory limits and trigger OOM errors.

### Recommended configuration
- When worker CPU and memory quotas are unknown or constrained, start with conservative values such as `-@ 4 -m 512M`.
- Increase `-@` or `-m` only within the known memory budget of the execution environment, leaving headroom for I/O decompression/compression.
- Lowering `-m` spills more intermediate runs to temporary files; ensure the scratch directory has adequate disk space for large alignments.
```

**避免写法**：只复盘“本次命令用了 `-@ 32 -m 2G` 导致 OOM，已改成 `-@ 4 -m 512M` 但尚未重跑验证”。

## 提交与确认流程

调用 `write_node_memory` 后，系统会把每条候选的完整 `description` 与 `content` 交由用户逐条审阅，仅保存用户明确同意的条目：

- 若用户拒绝某条建议，不得原样重复提交。
- 若用户给出修改意见，须按反馈修改后重新调用工具请求确认。
