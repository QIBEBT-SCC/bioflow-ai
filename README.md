# BioFlowAI: Graph-based Intelligent BioAnalysis Platform
## 测试数据

测试数据集托管在 Hugging Face，使用 `huggingface_hub` 或 `git` 下载到项目根目录的 `testdata/` 文件夹：

**方式一：使用 huggingface-cli（推荐）**
```bash
huggingface-cli download Aye10032/bioflow-ai-testdata --repo-type dataset --local-dir testdata
```

**方式二：使用 git clone**
```bash
git clone https://huggingface.co/datasets/Aye10032/bioflow-ai-testdata testdata
# 或使用 SSH
git clone git@hf.co:datasets/Aye10032/bioflow-ai-testdata testdata
```

## 初始化podman服务
```bash
podman system service --time=0 unix:///run/user/$(id -u)/podman/podman.sock > /dev/null 2>&1 &
```