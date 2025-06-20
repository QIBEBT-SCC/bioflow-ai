# BioFlowAI: Graph-based Intelligent BioAnalysis Platform
## 初始化podman服务
```bash
podman system service --time=0 unix:///run/user/$(id -u)/podman/podman.sock > /dev/null 2>&1 &
```