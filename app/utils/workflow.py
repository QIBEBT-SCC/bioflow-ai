from pathlib import Path

from app.models.react_flow import WorkflowDefinition


def run_from_json(json_str: str):
    workflow = WorkflowDefinition.model_validate_json(json_str)
    task_list = workflow.get_graph_list()
    for task in task_list:
        print(task)


def run_from_file(json_file: str | Path):
    with open(json_file, 'r', encoding='utf-8') as f:
        json_str = f.read()

    return run_from_json(json_str)


if __name__ == '__main__':
    run_from_file('../../storage/yeyu/workflow/default_workflow.json')
