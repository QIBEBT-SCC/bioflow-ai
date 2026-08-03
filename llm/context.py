from pydantic import BaseModel, ConfigDict


class BaseContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: int
    verbose: bool = False


class CreateToolContext(BaseContext):
    test_run_id: str

    auto_save: bool = True
    skip_search: bool = False

    save_message: bool = True


class TestRunContext(BaseContext):
    test_run_id: str

    save_message: bool = True


class ToolGeneratorContext(BaseContext):
    test_run_id: str

    save_message: bool = True


class ProjectContext(BaseContext):
    project_id: int
    workdir: str
    auto_approve: bool = False
    save_message: bool = True


class FolderContext(BaseContext):
    output_path: str
