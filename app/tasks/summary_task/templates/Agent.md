# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 环境管理：pixi（唯一指定工具）

**所有环境操作必须通过 pixi 完成，禁止使用 conda、pip、venv 或 mamba 直接管理依赖。**

### 核心原则

- 所有依赖写入 `pixi.toml`，通过 `pixi add` 添加，**不要手动编辑依赖版本字段**
- `pixi.lock` 必须提交到版本库，保证跨机器可复现
- `.pixi/` 目录已在 `.gitignore` 中，不提交
- 按分析类型划分 pixi feature/environment，避免单一环境过于臃肿

## 项目结构

```
project/
├── pixi.toml           # 依赖声明（提交）
├── pixi.lock           # 锁定版本（提交）
├── CLAUDE.md
├── data/
│   ├── raw/            # 原始输入文件（不提交大文件）
│   └── processed/      # 中间处理结果
├── results/            # 最终输出（图表、表格）
├── analysis/
│   ├── 01_qc.py / .R
│   ├── 02_normalization.py / .R
│   └── ...
└── docs/
    ├── methods.md      # 分析方法记录
    └── results.md      # 结果解读
```

## 分析规范

### 可视化

- 图表输出到 `results/figures/`，同时保存 PDF 和 PNG
- 配色方案：连续变量用 `viridis`，分类变量避免使用红绿组合（色盲友好）
- 所有图表须包含轴标签、图注和样本 N 数

### 代码规范

- Python：格式化工具为 `ruff`（通过 pixi 安装）
- R：`styler` + `lintr`
- 脚本顶部须声明：输入文件路径、输出文件路径、关键参数
- 使用相对路径，不硬编码绝对路径