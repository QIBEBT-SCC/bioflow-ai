from pathlib import Path

from fastapi import APIRouter

from app.schema.react_flow import WorkflowDefinition
from app.utils.config import get_settings

router = APIRouter()


@router.get('/flows')
async def get_workflows() -> list[str]:
    user = 'yeyu'

    wf_store_path = Path(get_settings().persistent.STORAGE_PATH) / user / 'workflow'
    wf_store_path.mkdir(parents=True, exist_ok=True)

    files = wf_store_path.glob('*.json')
    flows = [file.stem for file in files]
    return flows


@router.get('/flows/{name}')
async def get_workflow(name: str) -> WorkflowDefinition:
    user = 'yeyu'

    wf_store_path = Path(get_settings().persistent.STORAGE_PATH) / user / 'workflow'
    wf_store_path.mkdir(parents=True, exist_ok=True)

    with open(wf_store_path / f'{name}.json', 'r') as f:
        flow_str = f.read()

    workflow = WorkflowDefinition.model_validate_json(flow_str)
    return workflow


@router.post('/flows/{name}')
async def save_workflow(name: str, workflow: WorkflowDefinition) -> None:
    """保存工作流

    Args:
        name: 此工作流的名称
        workflow: 工作流定义，包含节点和边的信息
    """

    user = 'yeyu'

    wf_store_path = Path(get_settings().persistent.STORAGE_PATH) / user / 'workflow'
    wf_store_path.mkdir(parents=True, exist_ok=True)

    with open(wf_store_path / f'{name}.json', 'w') as f:
        f.write(workflow.model_dump_json())
