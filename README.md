<p align="center">
  <img src="https://raw.githubusercontent.com/QIBEBT-SCC/bioflow-ai-web/refs/heads/master/public/logo_and_text.svg" alt="Logo" width="500px" />
</p>

# BioFlowAI: Graph-based Intelligent BioAnalysis Platform

## Installation

### Prerequisites

This project uses [uv](https://docs.astral.sh/uv/) as the package manager. Install it before proceeding.

### 1. Clone and Install Dependencies

```bash
git clone https://github.com/QIBEBT-SCC/bioflow-ai.git
cd bioflow-ai
uv sync
```

### 2. Download Test Data

The test dataset is hosted on [Hugging Face](https://huggingface.co/buckets/Aye10032/bioflow-ai-testdata). Download it into the `testdata/` directory at the project root:

```bash
uvx hf sync hf://buckets/Aye10032/bioflow-ai-testdata ./testdata
```

### 3. Prepare Storage Volumes

Create the mount directory for RustFS and set the required permissions:

```bash
mkdir -p ./volumes/rustfs
chown -R 10001:10001 ./volumes/rustfs
```

### 4. Start Docker Services

Launch the database and other dependencies via Docker Compose:

```bash
docker compose up -d
```

### 5. Initialize the Database

Run the database initialization command:

```bash
uv run bioflow-ai init system-init
```

### 6. Start the Application

Start the Celery worker and the API server in separate terminals:

```bash
uv run bioflow-ai api
uv run bioflow-ai worker
```

The web interface is now accessible at `http://<host>:3001`.

---

## Using Podman

If you prefer Podman as the container runtime, follow these steps instead of the Docker steps above.

**Install with Podman support:**

```bash
uv sync --extra podman
```

**Start the Podman socket service:**

```bash
podman system service --time=0 unix:///run/user/$(id -u)/podman/podman.sock > /dev/null 2>&1 &
```

**Start the containers:**

```bash
podman-compose up -d
```



