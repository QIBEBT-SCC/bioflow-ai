from fastapi import APIRouter

from app.bioinfo.tool_info import DefaultArgs
from app.core.session import SessionDep
from app.models.tools import DockerToolCreate

router = APIRouter()


@router.post('')
def add_new_tool(tool_data: DockerToolCreate, session: SessionDep):
    print(tool_data.to_tool_data())


@router.get('/args')
def get_default_args() -> DefaultArgs:
    default_args = DefaultArgs()
    return default_args
