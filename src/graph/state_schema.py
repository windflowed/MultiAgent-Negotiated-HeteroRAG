"""
全局状态 Schema 定义
定义 AgentState 类型，规范所有字段的名称、类型、含义
覆盖原始输入、路由结果、检索结果、融合中间结果、最终输出全流程
"""

from typing import TypedDict, Annotated, List, Optional, Dict, Any
import operator


class AgentState(TypedDict):
    """
    Agent 工作流状态 Schema

    字段说明：
    1. 原始输入：用户查询
    2. 路由结果：激活的数据源、路由原因、路由置信度
    3. 检索结果：SQL/文档/知识图谱三路检索结果
    4. 融合中间结果：融合上下文、三元组、冲突、数据源统计
    5. 最终输出：答案、引用来源、最终置信度
    6. 错误处理：错误信息
    """

    # ===== 原始输入 =====
    query: str

    # ===== 路由结果 =====
    routed_sources: List[str]
    routing_reason: str
    routing_confidence: float

    # ===== 检索结果 =====
    sql_results: Optional[Dict[str, Any]]
    doc_results: Optional[Dict[str, Any]]
    kg_results: Optional[Dict[str, Any]]

    # ===== 融合中间结果 =====
    fused_context: str
    triples: List[Dict[str, Any]]
    conflicts: List[Dict[str, Any]]
    source_stats: Dict[str, int]

    # ===== 最终输出 =====
    answer: str
    sources: Annotated[List[str], operator.add]
    final_confidence: float

    # ===== 错误处理 =====
    error: Optional[str]


class SimpleAgentState(TypedDict):
    """
    简化版 Agent 工作流状态 Schema
    用于简单的测试场景
    """

    query: str
    routed_sources: List[str]
    sql_results: Optional[Dict[str, Any]]
    doc_results: Optional[Dict[str, Any]]
    kg_results: Optional[Dict[str, Any]]
    fused_context: str
    answer: str
    sources: Annotated[List[str], operator.add]


def create_initial_state(query: str) -> AgentState:
    """
    创建初始状态

    Args:
        query: 用户查询

    Returns:
        初始化的状态字典
    """
    return {
        "query": query,
        "routed_sources": [],
        "routing_reason": "",
        "routing_confidence": 0.0,
        "sql_results": None,
        "doc_results": None,
        "kg_results": None,
        "fused_context": "",
        "triples": [],
        "conflicts": [],
        "source_stats": {"sql": 0, "doc": 0, "kg": 0},
        "answer": "",
        "sources": [],
        "final_confidence": 0.0,
        "error": None
    }


def create_simple_initial_state(query: str) -> SimpleAgentState:
    """
    创建简化版初始状态

    Args:
        query: 用户查询

    Returns:
        初始化的状态字典
    """
    return {
        "query": query,
        "routed_sources": [],
        "sql_results": None,
        "doc_results": None,
        "kg_results": None,
        "fused_context": "",
        "answer": "",
        "sources": []
    }


if __name__ == "__main__":
    print("=" * 60)
    print("AgentState Schema 定义测试")
    print("=" * 60)

    # 测试初始状态创建
    initial_state = create_initial_state("阿凡达电影的评分是多少？")
    print("初始状态:")
    for key, value in initial_state.items():
        print(f"  {key}: {type(value).__name__} = {value}")

    print("\n简化版初始状态:")
    simple_state = create_simple_initial_state("阿凡达电影的评分是多少？")
    for key, value in simple_state.items():
        print(f"  {key}: {type(value).__name__} = {value}")
