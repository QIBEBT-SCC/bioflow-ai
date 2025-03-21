from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from app.schema.react_flow import WorkflowDefinition
from app.bioinfo.workflow_builder import WorkflowBuilder

router = APIRouter()


@router.post('/run')
async def run_workflow(workflow: WorkflowDefinition) -> Dict[str, Any]:
    """运行工作流

    Args:
        workflow: 工作流定义，包含节点和边的信息

    Returns:
        工作流执行结果
    """
    print(workflow)
    return {'status': 'success', 'result': 'OK'}
    # try:
    #     # 创建工作流构建器
    #     builder = WorkflowBuilder(
    #         nodes=[node.model_dump() for node in workflow.nodes],
    #         edges=[edge.model_dump() for edge in workflow.edges],
    #     )
    #
    #     # 构建并运行工作流
    #     flow = builder.build_flow()
    #     result = await flow()
    #
    #     return {'status': 'success', 'result': result}
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))


@router.post('/workflow/validate')
async def validate_workflow(workflow: WorkflowDefinition) -> Dict[str, Any]:
    """验证工作流定义是否有效

    Args:
        workflow: 工作流定义，包含节点和边的信息

    Returns:
        验证结果
    """
    try:
        # 检查节点连接是否完整
        node_ids = {node.id for node in workflow.nodes}
        for edge in workflow.edges:
            if edge.source not in node_ids:
                raise ValueError(f'源节点 {edge.source} 不存在')
            if edge.target not in node_ids:
                raise ValueError(f'目标节点 {edge.target} 不存在')

        # 检查输入输出端口是否匹配
        for node in workflow.nodes:
            # 检查输入端口
            for edge in workflow.edges:
                if edge.target == node.id:
                    target_handle = edge.targetHandle.split('-')[-1]
                    if not hasattr(node.data, f'{target_handle}Description'):
                        raise ValueError(f'节点 {node.id} 的输入端口 {target_handle} 不存在')

            # 检查输出端口
            for edge in workflow.edges:
                if edge.source == node.id:
                    source_handle = edge.sourceHandle.split('-')[-1]
                    if not hasattr(node.data, f'{source_handle}Description'):
                        raise ValueError(f'节点 {node.id} 的输出端口 {source_handle} 不存在')

        return {'status': 'success', 'message': '工作流定义有效'}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
