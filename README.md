# BioFlowAI: Graph-based Intelligent BioAnalysis Platform

## 测试数据

测试数据集托管在 [Hugging Face](https://huggingface.co/buckets/Aye10032/bioflow-ai-testdata)，使用 `huggingface_hub` 下载到项目根目录的 `testdata/` 文件夹：

```bash
uvx hf sync hf://buckets/Aye10032/bioflow-ai-testdata ./testdata
```

## 初始化podman服务

```bash
podman system service --time=0 unix:///run/user/$(id -u)/podman/podman.sock > /dev/null 2>&1 &
```