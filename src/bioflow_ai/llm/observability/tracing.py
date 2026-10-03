import os


def configure_langsmith_tracing(project: str, *, enabled: bool = False) -> None:
    """Configure LangSmith tracing explicitly for an agent command."""
    tracing_value = str(enabled).lower()
    os.environ['LANGSMITH_TRACING'] = tracing_value
    os.environ['LANGSMITH_TRACING_V2'] = tracing_value
    os.environ['LANGCHAIN_TRACING_V2'] = tracing_value
    os.environ['LANGSMITH_PROJECT'] = project
    os.environ['LANGCHAIN_PROJECT'] = project
