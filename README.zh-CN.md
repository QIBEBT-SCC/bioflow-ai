<p align="center">
  <img src="https://raw.githubusercontent.com/QIBEBT-SCC/bioflow-ai-web/refs/heads/master/public/logo_and_text.svg" alt="BioFlow AI" width="400" />
</p>

<p align="center">
  <a href="https://github.com/QIBEBT-SCC/bioflow-ai/actions/workflows/pr-ci.yml"><img src="https://github.com/QIBEBT-SCC/bioflow-ai/actions/workflows/pr-ci.yml/badge.svg?branch=master" alt="Backend CI" /></a>
  <a href="https://pypi.org/project/bioflow-ai/"><img src="https://img.shields.io/pypi/v/bioflow-ai" alt="PyPI version" /></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.13%2B-blue" alt="Python 3.13+" /></a>
  <a href="https://bioflow-ai-doc.vercel.app/"><img src="https://img.shields.io/badge/docs-bioflow--ai-informational" alt="Documentation" /></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0-blue" alt="License: AGPL-3.0" /></a>
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

在空的部署目录中从 PyPI 安装 `bioflow-ai`：

```bash
uv init --bare --python 3.13
uv add bioflow-ai
uv run bioflow-ai auto-init
```

如需安装特定的预发布版本，可显式指定版本号（例如 `uv add 'bioflow-ai==1.0.0b1'`）；如需离线部署，也可从 [GitHub Releases](https://github.com/QIBEBT-SCC/bioflow-ai/releases) 下载 `.whl` 文件并将其路径传给 `uv add`。使用 Podman 时，在安装命令中加入 `--extra podman`。初始化和三个后端进程都应在同一个部署目录中运行。初始化会把可编辑的 Skills 放到 `presets/skills/`，把数据库初始化文件放到 `presets/sql/`。再次运行时会补齐缺失的文件，并保留已有副本。

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

### 启动与管理后端

自动初始化程序先准备依赖服务和数据库表，再集中配置 Embedding 模型和聊天模型；随后创建管理员账号、导入系统预置数据、构建向量索引并可选下载测试数据，最后会直接询问是否在后台一键启动 API（内置工作流 Dispatcher）、统一执行 Worker 和智能体 Worker。参阅[自动初始化指南](https://bioflow-ai-doc.vercel.app/docs/configuration/initialization)。

后续可在同一部署目录中使用以下命令管理后台服务：

```bash
uv run bioflow-ai start       # 后台启动全部服务（也可传入指定服务名）
uv run bioflow-ai status      # 查看各服务运行状态、运行时长和 API 健康状态
uv run bioflow-ai logs -f     # 实时查看日志（也可指定 api / execution-worker / agent-worker）
uv run bioflow-ai restart     # 重启全部或单个服务（例如 uv run bioflow-ai restart api）
uv run bioflow-ai stop        # 停止全部或单个服务
```

Worker 并发数可在 `config.toml` 的 `[celery]` 段中通过 `execution_concurrency`（默认 `4`）和 `agent_concurrency`（默认 `4`）调整。

默认在浏览器中访问 `http://localhost:3000`。如果默认端口已被占用，自动初始化程序会选择可用端口并在终端中显示。

## 参与贡献

开发环境配置、分支规范与发版流程请参阅 [CONTRIBUTING.md](./CONTRIBUTING.md)。

## 开源协议

BioFlow AI 基于 [GNU Affero General Public License v3.0 (AGPL-3.0)](./LICENSE) 开源。
