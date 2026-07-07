from dataclasses import dataclass


@dataclass
class BaseContext:
    user_id: int


@dataclass
class CreateToolContext(BaseContext):
    auto_save: bool = True
    skip_search: bool = False

    save_message: bool = False
    dataset_file: str = 'test_run.parquet'
    test_run_id: str | None = None


@dataclass
class ProjectContext(BaseContext):
    project_id: int
    workdir: str
    auto_approve: bool = False
