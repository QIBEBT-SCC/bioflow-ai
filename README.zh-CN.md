<p align="center">
  <img src="https://raw.githubusercontent.com/QIBEBT-SCC/bioflow-ai-web/refs/heads/master/public/logo_and_text.svg" alt="BioFlow AI" width="400" />
</p>

<p align="center">
  <a href="./README.md">English</a> | 简体中文
</p>

# BioFlow AI

BioFlow AI 是一个图驱动的智能生物信息分析平台。它通过可视化节点工作流组织分析工具、数据和计算任务，并使用 AI 智能体辅助管理样本、构建工作流、诊断运行问题和修复分析流程。

## 系统依赖

- Linux（推荐）
- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- 以下容器运行时任选其一：
  - Docker + Docker Compose 插件
  - Podman + Podman Compose

推荐使用 rootless 容器运行时，避免分析任务生成的文件归属于宿主机 root 用户。

## 安装与初始化

在空的部署目录中安装对应版本的 wheel：

```bash
uv init --bare --python 3.13
uv add /path/to/bioflow_ai-1.0.0b1-py3-none-any.whl
uv run bioflow-ai auto-init
```

如果配置的包索引已提供 BioFlowAI，可以用 `uv add bioflow-ai` 安装。使用 Podman 时，在安装命令中加入
`--extra podman`。初始化和三个后端进程都应在同一个部署目录中运行。初始化会把可编辑的 Skills 放到
`presets/skills/`，把数据库初始化文件放到 `presets/sql/`。再次运行时会补齐缺失的文件，并保留已有副本。

### 从源码安装

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

### 启动后端

自动初始化程序会引导完成系统配置、测试数据、依赖容器、数据库、管理员账号和模型 Provider 的初始化。每项提示、可填写内容和默认值请参阅[自动初始化指南](https://bioflow-ai-doc.vercel.app/docs/configuration/initialization)。

初始化完成后，分别在独立终端中启动 API（内置工作流 Dispatcher）、统一执行 Worker 和智能体 Worker：

```bash
uv run bioflow-ai api
uv run bioflow-ai execution-worker --concurrency 4
uv run bioflow-ai agent-worker
```

默认在浏览器中访问 `http://localhost:3000`。如果默认端口已被占用，自动初始化程序会选择可用端口并在终端中显示。
