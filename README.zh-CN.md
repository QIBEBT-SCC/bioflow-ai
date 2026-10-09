<p align="center">
  <img src="https://raw.githubusercontent.com/QIBEBT-SCC/bioflow-ai-web/refs/heads/master/public/logo_and_text.svg" alt="BioFlow AI" width="400" />
</p>

<p align="center">
  <a href="./README.md">English</a> | 简体中文
</p>

# BioFlow AI

BioFlow AI 是一个图驱动的智能生物信息分析平台。它通过可视化节点工作流组织分析工具、数据和计算任务，并使用 AI 智能体辅助管理样本、构建工作流、诊断运行问题和修复分析流程。

平台由 FastAPI 后端、Celery 任务队列、AI 智能体、容器化分析工具和 [Web 前端](https://github.com/QIBEBT-SCC/bioflow-ai-web) 组成，适用于 RNA-seq、WGS、单细胞等生物信息分析场景。

## 系统依赖

- Linux（推荐）
- [Git](https://git-scm.com/)
- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- 以下容器运行时任选其一：
  - Docker + Docker Compose 插件
  - Podman + Podman Compose

推荐使用 rootless 容器运行时，避免分析任务生成的文件归属于宿主机 root 用户。

## 安装与初始化

```bash
git clone https://github.com/QIBEBT-SCC/bioflow-ai.git
cd bioflow-ai
```

使用 Docker：

```bash
uv sync
uv run bioflow-ai auto-init
```

使用 Podman：

```bash
uv sync --extra podman
uv run bioflow-ai auto-init
```

自动初始化程序会引导完成系统配置、测试数据、依赖容器、数据库、管理员账号和模型 Provider 的初始化。每项提示、可填写内容和默认值请参阅[自动初始化指南](https://github.com/QIBEBT-SCC/bioflow-ai-doc/blob/master/content/docs/configuration/initialization.zh.mdx)。

初始化完成后，分别在独立终端中启动 API、工作流 Dispatcher、统一执行 Worker 和智能体 Worker：

```bash
uv run bioflow-ai api
uv run bioflow-ai workflow-dispatcher
uv run bioflow-ai execution-worker --concurrency 4
uv run bioflow-ai agent-worker
```

默认在浏览器中访问 [http://localhost:3000](http://localhost:3000)。如果默认端口已被占用，自动初始化程序会选择可用端口并在终端中显示。

## 后端开发

Python 代码统一放在可安装的 `src/bioflow_ai` 包中。API、模型、Schema 和 CRUD 保留现有分层，业务服务按职责归组。进程入口位于 `runtime`，终端交互位于 `cli`，共享连接和容器支持位于 `infrastructure`，Agent 与工作流生成代码位于 `llm`。

```bash
uv run python -m pytest
uv run python -m ruff check src tests
uv run python -m pyrefly check
uv build
```
