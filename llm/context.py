from pydantic import BaseModel, ConfigDict


class BaseContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: int
    language: str | None = None
    verbose: bool = False
    delegate_human_requests: bool = False


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


class WorkflowDiagnosisContext(ProjectContext):
    agent_run_uid: str
    source_run_uid: str
    workflow_uid: str | None = None
    run_base_dir: str | None = None


class WorkflowFixContext(ProjectContext):
    agent_run_uid: str
    source_run_uid: str
    workflow_uid: str
    diagnosis_run_uid: str


class FolderContext(BaseContext):
    output_path: str
