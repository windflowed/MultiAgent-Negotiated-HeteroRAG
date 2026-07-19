"""
工作流节点函数封装
将 6 个 Agent 封装为 LangGraph 节点函数，输入 state，输出更新后的 state 字典
包含 StateGraph 构建与编译
"""

from typing import Dict, Any, List

from langgraph.graph import StateGraph, END, START

from src.graph.state_schema import AgentState
from src.agents.router_agent import create_router_agent
from src.agents.sql_agent import create_sql_agent_instance
from src.agents.doc_retriever import create_doc_retriever_agent
from src.agents.kg_agent import create_kg_agent
from src.agents.fusion_agent import create_fusion_agent
from src.agents.answer_generator import create_answer_generator


def router_node(state: AgentState) -> Dict[str, Any]:
    """
    路由节点：根据查询决定激活的数据源

    Args:
        state: 当前工作流状态

    Returns:
        更新后的状态字典
    """
    query = state["query"]
    router_agent = create_router_agent()

    try:
        result = router_agent.route(query)
        return {
            "routed_sources": result.get("sources", ["sql", "doc", "kg"]),
            "routing_reason": result.get("reason", ""),
            "routing_confidence": result.get("confidence", 0.0)
        }
    except Exception as e:
        return {
            "routed_sources": ["sql", "doc", "kg"],
            "routing_reason": f"路由失败: {str(e)}",
            "routing_confidence": 0.0,
            "error": str(e)
        }


def sql_retriever_node(state: AgentState) -> Dict[str, Any]:
    """
    SQL 检索节点：根据路由结果决定是否执行 SQL 查询

    Args:
        state: 当前工作流状态

    Returns:
        更新后的状态字典
    """
    if "sql" not in state.get("routed_sources", []):
        return {"sql_results": None}

    query = state["query"]
    sql_agent = create_sql_agent_instance()

    try:
        result = sql_agent.query(query)
        return {"sql_results": result}
    except Exception as e:
        return {
            "sql_results": {
                "success": False,
                "data": None,
                "error": str(e)
            }
        }


def doc_retriever_node(state: AgentState) -> Dict[str, Any]:
    """
    文档检索节点：根据路由结果决定是否执行文档检索

    Args:
        state: 当前工作流状态

    Returns:
        更新后的状态字典
    """
    if "doc" not in state.get("routed_sources", []):
        return {"doc_results": None}

    query = state["query"]
    doc_agent = create_doc_retriever_agent(use_reranker=False)

    try:
        result = doc_agent.retrieve(query, top_k=5)
        return {"doc_results": result}
    except Exception as e:
        return {
            "doc_results": {
                "success": False,
                "results": [],
                "error": str(e)
            }
        }


def kg_retriever_node(state: AgentState) -> Dict[str, Any]:
    """
    知识图谱检索节点：根据路由结果决定是否执行知识图谱查询

    Args:
        state: 当前工作流状态

    Returns:
        更新后的状态字典
    """
    if "kg" not in state.get("routed_sources", []):
        return {"kg_results": None}

    query = state["query"]
    kg_agent = create_kg_agent()

    try:
        result = kg_agent.query(query)
        return {"kg_results": result}
    except Exception as e:
        return {
            "kg_results": {
                "success": False,
                "triples": [],
                "error": str(e)
            }
        }


def fusion_node(state: AgentState) -> Dict[str, Any]:
    """
    融合节点：执行多源检索结果融合

    Args:
        state: 当前工作流状态

    Returns:
        更新后的状态字典
    """
    query = state["query"]
    sql_results = state.get("sql_results")
    doc_results = state.get("doc_results")
    kg_results = state.get("kg_results")

    fusion_agent = create_fusion_agent()

    try:
        result = fusion_agent.fuse(query, sql_results, doc_results, kg_results)
        return {
            "fused_context": result.get("fused_context", ""),
            "triples": result.get("triples", []),
            "conflicts": result.get("conflicts", []),
            "source_stats": result.get("source_stats", {"sql": 0, "doc": 0, "kg": 0})
        }
    except Exception as e:
        return {
            "fused_context": "",
            "triples": [],
            "conflicts": [],
            "source_stats": {"sql": 0, "doc": 0, "kg": 0},
            "error": str(e)
        }


def generator_node(state: AgentState) -> Dict[str, Any]:
    """
    生成节点：基于融合上下文生成答案

    Args:
        state: 当前工作流状态

    Returns:
        更新后的状态字典
    """
    query = state["query"]
    fused_context = state.get("fused_context", "")
    source_stats = state.get("source_stats", {})

    answer_generator = create_answer_generator()

    try:
        result = answer_generator.generate(query, fused_context, source_stats)
        return {
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "final_confidence": result.get("confidence", 0.0)
        }
    except Exception as e:
        return {
            "answer": f"答案生成失败: {str(e)}",
            "sources": [],
            "final_confidence": 0.0,
            "error": str(e)
        }


def route_to_sources(state: AgentState) -> List[str]:
    """
    条件边路由函数：根据路由结果返回需要激活的节点列表

    Args:
        state: 当前工作流状态

    Returns:
        需要激活的节点名称列表
    """
    sources = state.get("routed_sources", ["sql", "doc", "kg"])
    nodes = []

    if "sql" in sources:
        nodes.append("sql_retriever")
    if "doc" in sources:
        nodes.append("doc_retriever")
    if "kg" in sources:
        nodes.append("kg_retriever")

    return nodes


def build_workflow():
    """
    构建 LangGraph 工作流

    Returns:
        编译后的工作流应用实例
    """
    # 创建 StateGraph
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("router", router_node)
    workflow.add_node("sql_retriever", sql_retriever_node)
    workflow.add_node("doc_retriever", doc_retriever_node)
    workflow.add_node("kg_retriever", kg_retriever_node)
    workflow.add_node("fusion", fusion_node)
    workflow.add_node("generator", generator_node)

    # 设置入口
    workflow.set_entry_point("router")

    # 条件边：根据路由结果动态激活检索节点
    workflow.add_conditional_edges(
        "router",
        route_to_sources,
        {
            "sql_retriever": "sql_retriever",
            "doc_retriever": "doc_retriever",
            "kg_retriever": "kg_retriever",
        }
    )

    # 汇聚边：所有检索节点完成后进入融合
    workflow.add_edge("sql_retriever", "fusion")
    workflow.add_edge("doc_retriever", "fusion")
    workflow.add_edge("kg_retriever", "fusion")

    # 顺序边：融合 -> 生成 -> 结束
    workflow.add_edge("fusion", "generator")
    workflow.add_edge("generator", END)

    # 编译
    app = workflow.compile()

    return app


def get_compiled_workflow():
    """
    获取编译后的工作流实例（单例模式）

    Returns:
        编译后的工作流应用实例
    """
    if not hasattr(get_compiled_workflow, "_instance"):
        get_compiled_workflow._instance = build_workflow()
    return get_compiled_workflow._instance


if __name__ == "__main__":
    print("=" * 60)
    print("工作流构建与编译测试")
    print("=" * 60)

    # 测试工作流构建
    print("\n构建工作流...")
    app = build_workflow()
    print(f"工作流编译成功: {app is not None}")

    # 测试工作流执行
    print("\n测试工作流执行...")
    from src.graph.state_schema import create_initial_state

    test_state = create_initial_state("阿凡达电影的评分是多少？")
    print(f"初始状态: query={test_state['query']}")

    try:
        result = app.invoke(test_state)
        print(f"\n工作流执行成功:")
        print(f"  路由结果: {result.get('routed_sources', [])}")
        print(f"  答案: {result.get('answer', '')[:100]}...")
        print(f"  来源: {result.get('sources', [])}")
        print(f"  置信度: {result.get('final_confidence', 0.0):.2f}")
    except Exception as e:
        print(f"工作流执行失败: {e}")
