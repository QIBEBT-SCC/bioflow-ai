from dataclasses import dataclass


@dataclass
class BaseContext:
    user_id: int


@dataclass
class ProjectContext(BaseContext):
    project_id: int
    workdir: str
