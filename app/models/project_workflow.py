import uuid
from datetime import UTC, datetime

from sqlmodel import Field, Relationship, SQLModel


class ProjectWorkflow(SQLModel, table=True):
    """项目和工作流模板的关联表
    
    用于将工作流模板"导入"到项目中，使其成为项目的可用工作流
    """
    __tablename__ = 'project_workflows'
    
    project_id: int = Field(foreign_key='projects.id', primary_key=True)
    workflow_uid: uuid.UUID = Field(foreign_key='workflows.uid', primary_key=True)
    import_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    enabled: bool = Field(default=True, description='是否启用该工作流')
