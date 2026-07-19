"""
基线系统实现
实现三组基线系统，用于与主系统进行对比评测：
1. 单模态RAG：仅使用文档检索，无路由/融合
2. 简单拼接多源RAG：固定三源，简单拼接结果
3. 无协商多AgentRAG：三源+基本融合，无冲突消解

所有基线使用与主系统相同的组件（BGE嵌入、KG消歧），确保对比公平
"""

import time
import json
from typing import Dict, Any, List, Optional

from src.graph.state_schema import AgentState, create_initial_state
from src.agents.doc_retriever import create_doc_retriever_agent
from src.agents.sql_agent import create_sql_agent_instance
from src.agents.kg_agent import create_kg_agent
from src.agents.fusion_agent import create_fusion_agent
from src.agents.answer_generator import create_answer_generator


class BaselineSingleSourceRAG:
    """
    基线1：单模态RAG（Single-Source RAG）
    仅使用文档检索作为唯一数据源，无路由、无融合、无冲突消解
    代表传统单源RAG系统
    """

    def __init__(self):
        self.doc_agent = create_doc_retriever_agent(use_reranker=False)
        self.generator = create_answer_generator()

    def run(self, query: str) -> Dict[str, Any]:
        """
        执行单源RAG查询

        Args:
            query: 用户查询

        Returns:
            查询结果字典，格式与主系统一致
        """
        start_time = time.time()

        # 初始化状态
        state = create_initial_state(query)
        state["routed_sources"] = ["doc"]

        # 1. 文档检索
        doc_results = None
        try:
            doc_results = self.doc_agent.retrieve(query, top_k=5)
            state["doc_results"] = doc_results
        except Exception as e:
            state["error"] = f"文档检索失败: {str(e)}"

        # 2. 构建上下文
        context_parts = []
        if doc_results and doc_results.get("success") and doc_results.get("results"):
            for i, doc in enumerate(doc_results["results"][:3]):
                context_parts.append(f"[DOC{i+1}] {doc.get('document', '')[:200]}")

        fused_context = "\n".join(context_parts)
        state["fused_context"] = fused_context

        # 3. 生成答案
        if fused_context:
            try:
                result = self.generator.generate(query, fused_context)
                state["answer"] = result.get("answer", "")
                state["sources"] = result.get("sources", [])
                state["final_confidence"] = result.get("confidence", 0.0)
            except Exception as e:
                state["answer"] = f"答案生成失败: {str(e)}"
        else:
            state["answer"] = "未检索到相关文档信息"

        # 计算响应时间
        state["response_time"] = time.time() - start_time

        return state


class BaselineSimpleConcatRAG:
    """
    基线2：简单拼接多源RAG（Simple Concatenation Multi-Source RAG）
    固定激活全部三个数据源，简单拼接结果
    无动态路由，无冲突检测与消解
    代表无动态路由的多源RAG系统
    """

    def __init__(self):
        self.sql_agent = create_sql_agent_instance()
        self.doc_agent = create_doc_retriever_agent(use_reranker=False)
        self.kg_agent = create_kg_agent()
        self.generator = create_answer_generator()

    def run(self, query: str) -> Dict[str, Any]:
        """
        执行简单拼接多源RAG查询

        Args:
            query: 用户查询

        Returns:
            查询结果字典，格式与主系统一致
        """
        start_time = time.time()

        # 初始化状态（固定激活全部数据源）
        state = create_initial_state(query)
        state["routed_sources"] = ["sql", "doc", "kg"]

        # 1. 并行检索三个数据源
        sql_results = None
        doc_results = None
        kg_results = None

        try:
            sql_results = self.sql_agent.query(query)
            state["sql_results"] = sql_results
        except Exception as e:
            state["error"] = f"SQL检索失败: {str(e)}"

        try:
            doc_results = self.doc_agent.retrieve(query, top_k=5)
            state["doc_results"] = doc_results
        except Exception as e:
            state["error"] = f"文档检索失败: {str(e)}"

        try:
            kg_results = self.kg_agent.query(query)
            state["kg_results"] = kg_results
        except Exception as e:
            state["error"] = f"KG检索失败: {str(e)}"

        # 2. 简单拼接所有结果（无冲突消解）
        context_parts = []

        # SQL结果
        if sql_results and sql_results.get("success") and sql_results.get("data"):
            context_parts.append("[SQL] 数据库查询结果:")
            for row in sql_results["data"][:3]:
                context_parts.append(f"  - {row}")

        # 文档结果
        if doc_results and doc_results.get("success") and doc_results.get("results"):
            context_parts.append("[DOC] 文档检索结果:")
            for doc in doc_results["results"][:3]:
                context_parts.append(f"  - {doc.get('document', '')[:150]}...")

        # KG结果
        if kg_results and kg_results.get("success") and kg_results.get("triples"):
            context_parts.append("[KG] 知识图谱三元组:")
            for triple in kg_results["triples"][:5]:
                context_parts.append(
                    f"  - {triple.get('subject')} --[{triple.get('predicate')}]--> {triple.get('object')}"
                )

        fused_context = "\n".join(context_parts)
        state["fused_context"] = fused_context

        # 3. 生成答案
        if fused_context:
            try:
                result = self.generator.generate(query, fused_context)
                state["answer"] = result.get("answer", "")
                state["sources"] = result.get("sources", [])
                state["final_confidence"] = result.get("confidence", 0.0)
            except Exception as e:
                state["answer"] = f"答案生成失败: {str(e)}"
        else:
            state["answer"] = "未检索到相关信息"

        state["response_time"] = time.time() - start_time

        return state


class BaselineNoNegotiationRAG:
    """
    基线3：无协商多Agent RAG（Multi-Agent RAG without Dynamic Routing）
    使用三个Agent检索，有基本融合（置信度加权），但无协商机制（无冲突检测与消解）
    代表有Agent但无协商机制的系统
    """

    def __init__(self):
        self.sql_agent = create_sql_agent_instance()
        self.doc_agent = create_doc_retriever_agent(use_reranker=False)
        self.kg_agent = create_kg_agent()
        self.fusion_agent = create_fusion_agent()
        self.generator = create_answer_generator()

    def run(self, query: str) -> Dict[str, Any]:
        """
        执行无协商多Agent RAG查询

        Args:
            query: 用户查询

        Returns:
            查询结果字典，格式与主系统一致
        """
        start_time = time.time()

        # 初始化状态（固定激活全部数据源，无动态路由）
        state = create_initial_state(query)
        state["routed_sources"] = ["sql", "doc", "kg"]

        # 1. 并行检索三个数据源
        sql_results = None
        doc_results = None
        kg_results = None

        try:
            sql_results = self.sql_agent.query(query)
            state["sql_results"] = sql_results
        except Exception as e:
            state["error"] = f"SQL检索失败: {str(e)}"

        try:
            doc_results = self.doc_agent.retrieve(query, top_k=5)
            state["doc_results"] = doc_results
        except Exception as e:
            state["error"] = f"文档检索失败: {str(e)}"

        try:
            kg_results = self.kg_agent.query(query)
            state["kg_results"] = kg_results
        except Exception as e:
            state["error"] = f"KG检索失败: {str(e)}"

        # 2. 基本融合（使用FusionAgent但禁用冲突消解）
        try:
            # 直接构建上下文，不进行冲突检测
            context_parts = []

            # SQL结果（带权重标签）
            if sql_results and sql_results.get("success") and sql_results.get("data"):
                for row in sql_results["data"][:3]:
                    for key, value in row.items():
                        if value is not None:
                            context_parts.append(
                                f"[SQL] {row.get('title', '未知')} - {key}: {value}"
                            )

            # 文档结果（带权重标签）
            if doc_results and doc_results.get("success") and doc_results.get("results"):
                for doc in doc_results["results"][:3]:
                    context_parts.append(
                        f"[DOC] {doc.get('document', '')[:200]}"
                    )

            # KG结果（带权重标签）
            if kg_results and kg_results.get("success") and kg_results.get("triples"):
                for triple in kg_results["triples"][:5]:
                    context_parts.append(
                        f"[KG] {triple.get('subject')} --[{triple.get('predicate')}]--> {triple.get('object')}"
                    )

            fused_context = "\n".join(context_parts)
            state["fused_context"] = fused_context

            # 简单统计各数据源数量
            state["source_stats"] = {
                "sql": len(sql_results.get("data", [])) if sql_results and sql_results.get("success") else 0,
                "doc": len(doc_results.get("results", [])) if doc_results and doc_results.get("success") else 0,
                "kg": len(kg_results.get("triples", [])) if kg_results and kg_results.get("success") else 0
            }

        except Exception as e:
            state["error"] = f"融合失败: {str(e)}"

        # 3. 生成答案
        if state.get("fused_context"):
            try:
                result = self.generator.generate(
                    query,
                    state["fused_context"],
                    state.get("source_stats", {})
                )
                state["answer"] = result.get("answer", "")
                state["sources"] = result.get("sources", [])
                state["final_confidence"] = result.get("confidence", 0.0)
            except Exception as e:
                state["answer"] = f"答案生成失败: {str(e)}"
        else:
            state["answer"] = "未检索到相关信息"

        state["response_time"] = time.time() - start_time

        return state


def load_test_dataset(dataset_path: str = "eval/test_dataset.json") -> List[Dict]:
    """
    加载评测数据集

    Args:
        dataset_path: 评测数据集路径

    Returns:
        问题列表
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("questions", [])


def run_baseline_evaluation(
    baseline_class,
    test_dataset: List[Dict],
    max_questions: int = None
) -> List[Dict]:
    """
    运行基线系统评测

    Args:
        baseline_class: 基线系统类
        test_dataset: 测试数据集
        max_questions: 最大测试问题数（None表示全部）

    Returns:
        评测结果列表
    """
    results = []
    questions_to_test = test_dataset[:max_questions] if max_questions else test_dataset

    for i, item in enumerate(questions_to_test):
        question_id = item.get("id", i + 1)
        question = item.get("question", "")
        expected_answer = item.get("expected_answer", "")
        category = item.get("category", "")

        print(f"\n测试问题 {question_id}: {question[:50]}...")

        try:
            # 运行基线系统
            baseline = baseline_class()
            result = baseline.run(question)

            results.append({
                "id": question_id,
                "question": question,
                "expected_answer": expected_answer,
                "category": category,
                "generated_answer": result.get("answer", ""),
                "response_time": result.get("response_time", 0.0),
                "confidence": result.get("final_confidence", 0.0),
                "sources": result.get("sources", []),
                "error": result.get("error")
            })

            print(f"  生成答案: {result.get('answer', '')[:100]}...")
            print(f"  响应时间: {result.get('response_time', 0.0):.2f}秒")

        except Exception as e:
            print(f"  测试失败: {str(e)}")
            results.append({
                "id": question_id,
                "question": question,
                "expected_answer": expected_answer,
                "category": category,
                "generated_answer": "",
                "response_time": 0.0,
                "confidence": 0.0,
                "sources": [],
                "error": str(e)
            })

    return results


def save_results(results: List[Dict], output_path: str):
    """
    保存评测结果

    Args:
        results: 评测结果列表
        output_path: 输出文件路径
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"评测结果已保存到: {output_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("基线系统评测")
    print("=" * 60)

    # 加载测试数据集
    print("\n加载测试数据集...")
    test_dataset = load_test_dataset()
    print(f"共 {len(test_dataset)} 个测试问题")

    # 测试单个问题验证基线系统可运行
    print("\n" + "=" * 60)
    print("基线1：单模态RAG（Single-Source RAG）")
    print("=" * 60)
    try:
        baseline1 = BaselineSingleSourceRAG()
        result1 = baseline1.run("阿凡达电影的评分是多少？")
        print(f"测试成功！答案: {result1.get('answer', '')[:100]}...")
    except Exception as e:
        print(f"测试失败: {str(e)}")

    print("\n" + "=" * 60)
    print("基线2：简单拼接多源RAG（Simple Concatenation RAG）")
    print("=" * 60)
    try:
        baseline2 = BaselineSimpleConcatRAG()
        result2 = baseline2.run("阿凡达电影的评分是多少？")
        print(f"测试成功！答案: {result2.get('answer', '')[:100]}...")
    except Exception as e:
        print(f"测试失败: {str(e)}")

    print("\n" + "=" * 60)
    print("基线3：无协商多Agent RAG（No Negotiation RAG）")
    print("=" * 60)
    try:
        baseline3 = BaselineNoNegotiationRAG()
        result3 = baseline3.run("阿凡达电影的评分是多少？")
        print(f"测试成功！答案: {result3.get('answer', '')[:100]}...")
    except Exception as e:
        print(f"测试失败: {str(e)}")

    print("\n" + "=" * 60)
    print("基线系统验证完成！")
    print("=" * 60)
