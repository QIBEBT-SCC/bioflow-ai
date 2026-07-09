from dataclasses import dataclass


@dataclass
class BaseContext:
    user_id: int


@dataclass
class CreateToolContext(BaseContext):
    test_run_id: str

    auto_save: bool = True
    skip_search: bool = False

    save_message: bool = False
    dataset_file: str = 'test_run.parquet'


@dataclass
class TestRunContext(BaseContext):
    test_run_id: str

    save_message: bool
    dataset_file: str


@dataclass
class ProjectContext(BaseContext):
    project_id: int
    workdir: str
    auto_approve: bool = False
