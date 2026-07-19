"""
工作流节点函数封装
将 6 个 Agent 封装为 LangGraph 节点函数，输入 state，输出更新后的 state 字典
"""

from typing import Dict, Any

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


def route_to_sources(state: AgentState) -> list:
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


if __name__ == "__main__":
    print("=" * 60)
    print("工作流节点函数测试")
    print("=" * 60)

    from src.graph.state_schema import create_initial_state

    # 测试初始状态
    test_state = create_initial_state("阿凡达电影的评分是多少？")
    print(f"初始状态: {test_state}")

    # 测试路由节点
    print("\n测试路由节点:")
    updated_state = router_node(test_state)
    print(f"更新后的状态: {updated_state}")

    # 测试 SQL 检索节点
    test_state.update(updated_state)
    print("\n测试 SQL 检索节点:")
    sql_result = sql_retriever_node(test_state)
    print(f"SQL 检索结果: {sql_result}")

    # 测试文档检索节点
    print("\n测试文档检索节点:")
    doc_result = doc_retriever_node(test_state)
    print(f"文档检索结果: {doc_result}")

    # 测试知识图谱检索节点
    print("\n测试知识图谱检索节点:")
    kg_result = kg_retriever_node(test_state)
    print(f"知识图谱检索结果: {kg_result}")
