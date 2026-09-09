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
- Python 3.12+
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

初始化完成后，分别在独立终端中启动 API、工作流 Dispatcher、工作流 Worker、资源 Worker 和智能体 Worker：

```bash
uv run bioflow-ai api
uv run bioflow-ai workflow-dispatcher
uv run bioflow-ai workflow-worker --concurrency 4
uv run bioflow-ai resource-worker --concurrency 2
uv run bioflow-ai agent-worker
```

如果现有环境已运行过旧的 `app/v2` 原型，请先备份 PostgreSQL，再执行[一次性工作流运行时迁移](presets/migrations/20260909_integrate_workflow_runtime.sql)。该迁移会保留已验证的 V2 运行表并重命名为最终表名，同时按计划删除 V1 的 Run/Task/Monitor 历史。

默认在浏览器中访问 [http://localhost:3000](http://localhost:3000)。如果默认端口已被占用，自动初始化程序会选择可用端口并在终端中显示。
