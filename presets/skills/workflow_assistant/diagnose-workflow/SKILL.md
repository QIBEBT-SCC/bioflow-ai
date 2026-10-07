---
name: diagnose-workflow
description: Diagnose or troubleshoot a workflow run when the user requests investigation of a failure or its root cause, and save an evidence-backed diagnosis report. Ordinary status questions, result explanations, and follow-up questions about an existing report do not require this skill or a new report.
---

# 工作流运行诊断

对当前指定运行调查故障原因，给出有证据支持的结论和具体修复建议。仅针对本轮的诊断请求执行此流程，历史诊断请求不构成新的报告要求。

## 调查

1. 调用 `workflow_reader(action="overview")`，定位直接失败节点及相关上游节点；按返回的分页信息继续读取需要的节点。区分节点自身失败、未运行和上游阻塞。
2. 普通子图按返回的 `child_path` 使用 `subgraph_path` 逐层进入。批量子图调用 `workflow_reader(action="children", node_id=..., subgraph_path=..., failed_only=true)`，分页读取失败项，再用返回的 `child_run_uid` 查看其运行。记录批量项和子图路径，区分子图内部失败与上游阻塞。
3. 对直接失败节点调用 `workflow_reader(action="node", subgraph_path=..., node_id=..., child_run_uid=...)`，其中 `node_id` 是所选层级的本地 ID。检查完整节点配置、实际命令、目录及可用的 Task/Celery error 和 traceback。
4. 用 `run_command` 检查对应运行目录：根运行使用 `run:current`，已确认的批量子运行使用 `run:child:<uid>`。先用 `find` 或 `file` 确认文件和类型，再检查失败节点目录里的 `11_run.log`、stderr/stdout、工具日志和相关小型 QC/结果文件。证据位于 `/app/data/input`，每次只检查一个来源。
5. 当节点配置或日志表明输入数据、格式或兼容性可能有关时，逐个检查已登记的 `sample:<tag>` 或 `proj:<keyword>`。不要把宿主机绝对路径作为 source。
6. 交叉验证工作流结构、命令/参数、输入数据、运行环境和日志。将事实与推断分开；命令失败、超时、截断或证据不可用时说明限制。证据不足就明确说明，不能把猜测当作确定根因，也不能因为当前没有失败就虚构故障。

## 报告提交

调查结束后必须调用 `write_diagnosis(content=...)`，提交一份自包含的 Markdown 报告。遵循系统指定的用户语言，未指定时用中文。报告包含以下内容，标题可随语言调整：

- **诊断结论**：故障结论及确定程度；无法确定或未发现失败时如实说明。
- **失败位置**：运行 UID、generation、子图路径、批量项（如有）、运行节点 ID、模板节点 ID（如有）、节点类型、Task/Celery 状态、执行目录和停止位置。
- **证据**：每条证据注明节点字段、Celery traceback、运行来源下的相对路径、`sample:<tag>` 或 `proj:<keyword>`，区分事实和推断。
- **根因分析**：解释证据如何支持结论，并列出尚未确认的因素。
- **修复建议**：给出后续工作流修复助手或人工可执行的具体修改。
- **影响范围与验证建议**：说明受影响的数据或步骤，以及验证修复的方法。

工具成功保存报告后，在聊天中简短说明结论和报告已保存，不重复报告正文。后续仅解释报告或回答普通问题时直接回答；用户再次请求诊断时才重新调查并写新报告。
