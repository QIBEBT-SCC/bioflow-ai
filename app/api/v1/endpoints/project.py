import uuid
from typing import Annotated, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from starlette.status import HTTP_404_NOT_FOUND

from app.core.session import SessionDep
from app.crud import project as project_crud
from app.crud import projetc_tag as tag_crud
from app.crud.resource import sample as sample_crud
from app.models.project import (
    Project,
    ProjectTag,
)
from app.models.resource import Sample, SampleFiles
from app.schemas.project import ProjectCreateProp, ProjectPublic, TagProp, TagWithCount
from app.schemas.resource import (
    SampleBatchImportResult,
    SampleCreate,
    SampleFileCreate,
    SampleFilePublic,
    SampleListItem,
    SamplePublic,
    SampleUpdate,
)
from app.utils.security import UserDep

router = APIRouter()
tag_router = APIRouter()


@tag_router.post('')
def new_tag(
    tag_prop: TagProp,
    session: SessionDep,
    current_user: UserDep,
):
    tag = ProjectTag(name=tag_prop.name, color=tag_prop.color)
    tag_crud.insert(session, tag)

    return tag


@tag_router.get('', response_model=list[TagWithCount])
def get_tag_list(session: SessionDep, current_user: UserDep):
    tags = tag_crud.get_all(session)
    return tags


@router.post('')
def new_project(
    project_prop: ProjectCreateProp,
    session: SessionDep,
    current_user: UserDep,
):
    tags = []
    if project_prop.tag_ids:
        tags = tag_crud.get_all_by_id(session, project_prop.tag_ids)

    project = Project(
        owner_id=current_user.id,
        name=project_prop.name,
        description=project_prop.description,
        public=project_prop.public,
        tags=tags,
    )
    project_crud.insert(session, project)
    return project


@router.post('/{project_id}/star')
def star_project(
    project_id: int,
    session: SessionDep,
    current_user: UserDep,
):
    project_crud.star_project(session, current_user.id, project_id)


@router.post('/{project_id}/unstar')
def unstar_project(
    project_id: int,
    session: SessionDep,
    current_user: UserDep,
):
    project_crud.unstar_project(session, current_user.id, project_id)


@router.get('', response_model=list[ProjectPublic])
def get_project_list(
    session: SessionDep,
    current_user: UserDep,
    filter_type: Annotated[Optional[str], Query(description='过滤条件: starred, my, all', alias='filter')] = 'all',
    offset: int = 0,
    limit: Annotated[int, Query(le=20)] = 20,
):
    """获取项目列表

    - filter=all: 获取所有可见项目（公开的或用户拥有的）
    - filter=starred: 获取用户收藏的项目
    - filter=my: 获取用户拥有的项目
    """
    match filter_type:
        case 'starred':
            return project_crud.get_all_star(session, limit, offset, current_user.id)
        case 'my':
            return project_crud.get_all_my(session, limit, offset, current_user.id)
        case 'all':
            return project_crud.get_all(session, limit, offset, current_user.id)
        case _:
            return []


@router.get('/recent', response_model=ProjectPublic)
def get_recent_project(session: SessionDep, current_user: UserDep):
    """获取用户最近更新的项目"""
    return project_crud.get_recent(session, current_user.id)


@router.get('/{project_id}', response_model=ProjectPublic)
def get_project(
    project_id: int,
    session: SessionDep,
    current_user: UserDep,
):
    return project_crud.get(session, current_user.id, project_id)


# ==================== 样本管理端点（在项目下） ====================


@router.post('/{project_id}/samples', response_model=SamplePublic)
async def create_sample_in_project(
    project_id: int,
    sample_prop: SampleCreate,
    session: SessionDep,
    current_user: UserDep,
):
    """在指定项目中创建新样本"""
    # 确保样本属于指定的项目
    if sample_prop.project_id != project_id:
        raise HTTPException(400, 'Project ID mismatch')

    sample = Sample(
        owner_id=current_user.id,
        project_id=project_id,
        sample_name=sample_prop.sample_name,
        meta_data=sample_prop.meta_data or {},
    )

    # 如果提供了文件信息，创建文件记录（文件信息由后端自动计算）
    files = []
    if sample_prop.files:
        for file_prop in sample_prop.files:
            # 只需要 data_type 和 file_path，其他信息由 CRUD 层自动计算
            sample_file = SampleFiles(
                sample_uid=sample.uid,
                data_type=file_prop.data_type,
                file_path=file_prop.file_path,
                file_format='',  # 将由 CRUD 自动计算
                file_size=0,  # 将由 CRUD 自动计算
                md5_checksum='',  # 将由 CRUD 自动计算
            )
            files.append(sample_file)

    sample_crud.insert(session, sample, files)
    return sample_crud.get_with_files(session, sample.uid)


@router.get('/{project_id}/samples', response_model=list[SampleListItem])
async def get_project_samples(
    project_id: int,
    session: SessionDep,
    current_user: UserDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=100)] = 20,
) -> list[SampleListItem]:
    """获取项目下的所有样本"""
    return sample_crud.get_all_by_project(session, project_id, limit, offset)


@router.get('/{project_id}/samples/count')
async def get_project_sample_count(
    project_id: int,
    session: SessionDep,
    current_user: UserDep,
) -> int:
    """获取项目下的样本数量"""
    return sample_crud.get_count_by_project(session, project_id)


@router.post('/{project_id}/samples/batch-import', response_model=SampleBatchImportResult)
async def batch_import_samples(
    project_id: int,
    session: SessionDep,
    current_user: UserDep,
    file: UploadFile = File(...),
) -> SampleBatchImportResult:
    """批量导入样本

    上传 CSV 文件批量创建样本。CSV 文件格式要求：
    - 编码：UTF-8
    - 必需列：sample_name
    - 可选文件列：r1, r2, single, spectrum, image（文件路径）
    - 其他列：自定义元数据，将存储在 meta_data 中

    示例 CSV：
    ```
    sample_name,condition,batch,r1,r2
    Sample-001,control,2024-01,/data/s1_R1.fq.gz,/data/s1_R2.fq.gz
    Sample-002,treatment,2024-01,/data/s2_R1.fq.gz,/data/s2_R2.fq.gz
    ```
    """
    # 验证文件类型
    if not file.filename or not file.filename.endswith('.csv'):
        raise HTTPException(400, '只支持 CSV 文件')

    # 读取文件内容
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(400, f'读取文件失败: {str(e)}')

    # 批量导入
    result = sample_crud.batch_import_from_csv(session, content, project_id, current_user.id)

    return result


@router.get('/{project_id}/samples/{sample_uid}', response_model=SamplePublic)
async def get_project_sample(
    project_id: int,
    sample_uid: uuid.UUID,
    session: SessionDep,
    current_user: UserDep,
):
    """获取项目下的指定样本详情"""
    sample = sample_crud.get_with_files(session, sample_uid)
    if not sample:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Sample not found')
    if sample.project_id != project_id:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Sample not found in this project')
    return sample


@router.put('/{project_id}/samples/{sample_uid}', response_model=SamplePublic)
async def update_project_sample(
    project_id: int,
    sample_uid: uuid.UUID,
    sample_update: SampleUpdate,
    session: SessionDep,
    current_user: UserDep,
):
    """更新项目下的样本信息"""
    existing_sample = sample_crud.get(session, sample_uid)
    if not existing_sample or existing_sample.project_id != project_id:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Sample not found in this project')

    sample_crud.update(session, sample_uid, sample_update)
    return sample_crud.get_with_files(session, sample_uid)


@router.delete('/{project_id}/samples/{sample_uid}')
async def delete_project_sample(
    project_id: int,
    sample_uid: uuid.UUID,
    session: SessionDep,
    current_user: UserDep,
):
    """删除项目下的样本"""
    existing_sample = sample_crud.get(session, sample_uid)
    if not existing_sample or existing_sample.project_id != project_id:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Sample not found in this project')

    sample_crud.delete(session, sample_uid)
    return {'message': 'Sample deleted successfully'}


@router.post('/{project_id}/samples/{sample_uid}/files', response_model=SampleFilePublic)
async def add_file_to_project_sample(
    project_id: int,
    sample_uid: uuid.UUID,
    file_prop: SampleFileCreate,
    session: SessionDep,
    current_user: UserDep,
):
    """为项目下的样本添加文件"""
    existing_sample = sample_crud.get(session, sample_uid)
    if not existing_sample or existing_sample.project_id != project_id:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Sample not found in this project')

    # 只需要提供 data_type 和 file_path，其他信息由后端自动计算
    sample_file = SampleFiles(
        sample_uid=sample_uid,
        data_type=file_prop.data_type,
        file_path=file_prop.file_path,
        file_format='',  # 将由 CRUD 自动计算
        file_size=0,  # 将由 CRUD 自动计算
        md5_checksum='',  # 将由 CRUD 自动计算
    )

    return sample_crud.add_file(session, sample_file)


@router.delete('/{project_id}/samples/{sample_uid}/files/{file_uid}')
async def delete_file_from_project_sample(
    project_id: int,
    sample_uid: uuid.UUID,
    file_uid: uuid.UUID,
    session: SessionDep,
    current_user: UserDep,
):
    """删除项目下样本的文件"""
    existing_sample = sample_crud.get(session, sample_uid)
    if not existing_sample or existing_sample.project_id != project_id:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Sample not found in this project')

    sample_crud.delete_file(session, file_uid)
    return {'message': 'File deleted successfully'}


@router.get('/{project_id}/samples/{sample_uid}/files', response_model=list[SampleFilePublic])
async def get_project_sample_files(
    project_id: int,
    sample_uid: uuid.UUID,
    session: SessionDep,
    current_user: UserDep,
):
    """获取项目下样本的所有文件"""
    existing_sample = sample_crud.get(session, sample_uid)
    if not existing_sample or existing_sample.project_id != project_id:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Sample not found in this project')

    return sample_crud.get_files(session, sample_uid)
