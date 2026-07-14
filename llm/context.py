from pydantic import BaseModel, ConfigDict


class BaseContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: int
    verbose: bool = False


class CreateToolContext(BaseContext):
    test_run_id: str

    auto_save: bool = True
    skip_search: bool = False

    save_message: bool = False
    dataset_file: str = 'test_run.parquet'


class TestRunContext(BaseContext):
    test_run_id: str

    save_message: bool
    dataset_file: str


class ToolGeneratorContext(BaseContext):
    test_run_id: str


class ProjectContext(BaseContext):
    project_id: int
    workdir: str
    auto_approve: bool = False


class FolderContext(BaseContext):
    output_path: str
