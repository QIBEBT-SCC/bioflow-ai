from fastapi import APIRouter

from app.bioinfo.tool_info import DefaultArgs

router = APIRouter()


@router.get('/args')
def get_default_args() -> DefaultArgs:
    default_args = DefaultArgs()
    return default_args
