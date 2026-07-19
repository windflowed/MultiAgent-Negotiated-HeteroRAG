"""
协商融合 Agent
实现三元组抽取、数值冲突检测、置信度加权表决三大核心逻辑
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from src.utils.common import load_config
from src.utils.prompts import (
    TRIPLE_EXTRACTION_PROMPT,
    CONFLICT_DETECTION_PROMPT,
    FUSION_WEIGHTED_PROMPT
)


class FusionAgent:
    """协商融合 Agent，实现多源检索结果的融合与冲突消解"""

    # 数据源基础权重
    SOURCE_WEIGHTS = {
        "sql": 0.5,
        "doc": 0.3,
        "kg": 0.2
    }

    def __init__(self, llm=None):
        """
        初始化协商融合 Agent

        Args:
            llm: 语言模型实例，为 None 时从配置创建
        """
        self.config = load_config()

        if llm is None:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model=self.config["LLM_MODEL_NAME"],
                api_key=self.config["LLM_API_KEY"],
                base_url=self.config["LLM_API_BASE"],
                temperature=0,
            )
        else:
            self.llm = llm

    def _extract_triples_from_text(self, text: str) -> List[Dict[str, str]]:
        """
        从文本中抽取三元组

        Args:
            text: 文本内容

        Returns:
            三元组列表
        """
        prompt = TRIPLE_EXTRACTION_PROMPT.format(text=text)

        try:
            response = self.llm.invoke(prompt)
            content = response.content

            # 提取 JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result.get("triples", [])
        except Exception as e:
            print(f"三元组抽取失败: {e}")

        return []

    def _extract_triples_from_results(
        self,
        sql_results: Optional[Dict] = None,
        doc_results: Optional[Dict] = None,
        kg_results: Optional[Dict] = None
    ) -> Dict[str, List[Dict]]:
        """
        从各数据源结果中抽取三元组

        Args:
            sql_results: SQL 检索结果
            doc_results: 文档检索结果
            kg_results: 知识图谱检索结果

        Returns:
            按数据源分组的三元组字典
        """
        all_triples = {"sql": [], "doc": [], "kg": []}

        # 从 SQL 结果抽取三元组
        if sql_results and sql_results.get("success") and sql_results.get("data"):
            for row in sql_results["data"][:5]:  # 限制数量
                for key, value in row.items():
                    if value is not None:
                        all_triples["sql"].append({
                            "subject": row.get("title", row.get("title_zh", "未知")),
                            "predicate": key,
                            "object": str(value),
                            "confidence": sql_results.get("confidence", 0.8),
                            "source": "sql"
                        })

        # 从文档结果抽取三元组
        if doc_results and doc_results.get("success") and doc_results.get("results"):
            for doc in doc_results["results"][:5]:
                triples = self._extract_triples_from_text(doc.get("document", ""))
                for triple in triples:
                    triple["confidence"] = doc.get("confidence", 0.7)
                    triple["source"] = "doc"
                all_triples["doc"].extend(triples)

        # 从知识图谱结果抽取三元组
        if kg_results and kg_results.get("success") and kg_results.get("triples"):
            for triple in kg_results["triples"][:10]:
                all_triples["kg"].append({
                    "subject": triple.get("subject", ""),
                    "predicate": triple.get("predicate", ""),
                    "object": triple.get("object", ""),
                    "confidence": triple.get("confidence", 0.8),
                    "source": "kg"
                })

        return all_triples

    def _detect_conflicts(
        self,
        triples1: List[Dict],
        triples2: List[Dict],
        source1: str,
        source2: str
    ) -> List[Dict[str, Any]]:
        """
        检测两组三元组之间的冲突

        Args:
            triples1: 第一组三元组
            triples2: 第二组三元组
            source1: 第一组来源
            source2: 第二组来源

        Returns:
            冲突列表
        """
        conflicts = []

        # 构建主体-属性 -> 值的映射
        map1 = {}
        for t in triples1:
            key = (t["subject"], t["predicate"])
            map1[key] = t

        map2 = {}
        for t in triples2:
            key = (t["subject"], t["predicate"])
            map2[key] = t

        # 检测冲突
        for key, t1 in map1.items():
            if key in map2:
                t2 = map2[key]
                value1 = t1["object"]
                value2 = t2["object"]

                # 检查数值冲突
                try:
                    num1 = float(re.sub(r'[^\d.]', '', str(value1)))
                    num2 = float(re.sub(r'[^\d.]', '', str(value2)))

                    if num1 != 0 and abs(num1 - num2) / abs(num1) > 0.05:
                        conflicts.append({
                            "subject": key[0],
                            "predicate": key[1],
                            "value1": value1,
                            "source1": source1,
                            "value2": value2,
                            "source2": source2,
                            "conflict_type": "numeric"
                        })
                except (ValueError, ZeroDivisionError):
                    # 文本冲突检测
                    if value1.lower() != value2.lower():
                        conflicts.append({
                            "subject": key[0],
                            "predicate": key[1],
                            "value1": value1,
                            "source1": source1,
                            "value2": value2,
                            "source2": source2,
                            "conflict_type": "textual"
                        })

        return conflicts

    def _calculate_final_confidence(
        self,
        triple: Dict[str, Any]
    ) -> float:
        """
        计算三元组的最终权重置信度

        Args:
            triple: 三元组字典

        Returns:
            最终权重置信度
        """
        source = triple.get("source", "doc")
        base_confidence = triple.get("confidence", 0.7)
        source_weight = self.SOURCE_WEIGHTS.get(source, 0.3)

        return source_weight * base_confidence

    def _resolve_conflicts(
        self,
        all_triples: List[Dict[str, Any]],
        conflicts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        解决冲突，保留置信度最高的结果

        Args:
            all_triples: 所有三元组
            conflicts: 冲突列表

        Returns:
            解决冲突后的三元组列表
        """
        if not conflicts:
            return all_triples

        # 构建需要移除的三元组集合
        to_remove = set()

        for conflict in conflicts:
            # 找到冲突的三元组
            matching_triples = []
            for i, triple in enumerate(all_triples):
                if (triple["subject"] == conflict["subject"] and
                    triple["predicate"] == conflict["predicate"]):
                    matching_triples.append((i, triple))

            if len(matching_triples) > 1:
                # 按置信度排序
                matching_triples.sort(
                    key=lambda x: self._calculate_final_confidence(x[1]),
                    reverse=True
                )

                # 保留置信度最高的，移除其他
                best_idx = matching_triples[0][0]
                for idx, _ in matching_triples[1:]:
                    to_remove.add(idx)

        # 移除冲突的三元组
        resolved = [
            triple for i, triple in enumerate(all_triples)
            if i not in to_remove
        ]

        return resolved

    def _build_context_with_sources(
        self,
        triples: List[Dict[str, Any]],
        sql_results: Optional[Dict] = None,
        doc_results: Optional[Dict] = None,
        kg_results: Optional[Dict] = None
    ) -> str:
        """
        构建带来源标签的上下文文本

        Args:
            triples: 融合后的三元组
            sql_results: SQL 检索结果
            doc_results: 文档检索结果
            kg_results: 知识图谱检索结果

        Returns:
            带来源标签的上下文文本
        """
        context_parts = []
        source_counter = {"sql": 1, "doc": 1, "kg": 1}

        # 添加三元组信息
        for triple in triples:
            source = triple.get("source", "doc")
            source_idx = source_counter[source]
            context_parts.append(
                f"[{source.upper()}{source_idx}] "
                f"{triple['subject']} - {triple['predicate']} - {triple['object']}"
            )
            source_counter[source] += 1

        # 添加原始结果摘要
        if sql_results and sql_results.get("data"):
            context_parts.append(f"\n[SQL] 数据库查询结果摘要:")
            for row in sql_results["data"][:3]:
                context_parts.append(f"  - {row}")

        if doc_results and doc_results.get("results"):
            context_parts.append(f"\n[DOC] 文档检索结果摘要:")
            for doc in doc_results["results"][:2]:
                context_parts.append(
                    f"  - {doc.get('document', '')[:100]}..."
                )

        if kg_results and kg_results.get("triples"):
            context_parts.append(f"\n[KG] 知识图谱三元组:")
            for triple in kg_results["triples"][:3]:
                context_parts.append(
                    f"  - {triple.get('subject')} --[{triple.get('predicate')}]--> {triple.get('object')}"
                )

        return "\n".join(context_parts)

    def fuse(
        self,
        query: str,
        sql_results: Optional[Dict] = None,
        doc_results: Optional[Dict] = None,
        kg_results: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        执行多源检索结果融合

        Args:
            query: 原始查询
            sql_results: SQL 检索结果
            doc_results: 文档检索结果
            kg_results: 知识图谱检索结果

        Returns:
            融合结果字典
        """
        result = {
            "success": False,
            "fused_context": "",
            "triples": [],
            "conflicts": [],
            "source_stats": {"sql": 0, "doc": 0, "kg": 0},
            "confidence": 0.0,
            "error": None
        }

        try:
            # 1. 从各数据源抽取三元组
            all_triples = self._extract_triples_from_results(
                sql_results, doc_results, kg_results
            )

            # 统计各数据源三元组数量
            for source, triples in all_triples.items():
                result["source_stats"][source] = len(triples)

            # 合并所有三元组
            merged_triples = []
            for source, triples in all_triples.items():
                merged_triples.extend(triples)

            if not merged_triples:
                # 如果没有三元组，构建基本上下文
                context = self._build_context_with_sources(
                    [], sql_results, doc_results, kg_results
                )
                result["success"] = True
                result["fused_context"] = context
                result["confidence"] = 0.5
                return result

            # 2. 检测冲突
            all_conflicts = []

            # SQL vs Doc
            if all_triples["sql"] and all_triples["doc"]:
                conflicts = self._detect_conflicts(
                    all_triples["sql"], all_triples["doc"],
                    "sql", "doc"
                )
                all_conflicts.extend(conflicts)

            # SQL vs KG
            if all_triples["sql"] and all_triples["kg"]:
                conflicts = self._detect_conflicts(
                    all_triples["sql"], all_triples["kg"],
                    "sql", "kg"
                )
                all_conflicts.extend(conflicts)

            # Doc vs KG
            if all_triples["doc"] and all_triples["kg"]:
                conflicts = self._detect_conflicts(
                    all_triples["doc"], all_triples["kg"],
                    "doc", "kg"
                )
                all_conflicts.extend(conflicts)

            result["conflicts"] = all_conflicts

            # 3. 解决冲突
            resolved_triples = self._resolve_conflicts(
                merged_triples, all_conflicts
            )

            # 4. 计算置信度
            if resolved_triples:
                confidences = [
                    self._calculate_final_confidence(t)
                    for t in resolved_triples
                ]
                result["confidence"] = sum(confidences) / len(confidences)

            # 5. 构建上下文
            context = self._build_context_with_sources(
                resolved_triples, sql_results, doc_results, kg_results
            )

            result["success"] = True
            result["fused_context"] = context
            result["triples"] = resolved_triples

        except Exception as e:
            result["error"] = str(e)

        return result


def create_fusion_agent() -> FusionAgent:
    """创建协商融合 Agent 实例"""
    return FusionAgent()


if __name__ == "__main__":
    print("=" * 60)
    print("协商融合 Agent 测试")
    print("=" * 60)

    agent = create_fusion_agent()

    # 模拟检索结果
    sql_results = {
        "success": True,
        "data": [{"title": "阿凡达", "vote_average": 7.9, "revenue": 2787965087}],
        "confidence": 0.9,
        "source": "sql"
    }

    doc_results = {
        "success": True,
        "results": [
            {"document": "阿凡达是一部科幻电影，由詹姆斯·卡梅隆执导。", "confidence": 0.8, "source": "doc"}
        ],
        "source": "doc"
    }

    kg_results = {
        "success": True,
        "triples": [
            {"subject": "阿凡达", "predicate": "属于", "object": "Action", "confidence": 0.8, "source": "kg"},
            {"subject": "阿凡达", "predicate": "主演", "object": "Sam Worthington", "confidence": 0.8, "source": "kg"}
        ],
        "source": "kg"
    }

    result = agent.fuse("阿凡达电影信息", sql_results, doc_results, kg_results)
    print(f"成功: {result['success']}")
    print(f"置信度: {result['confidence']:.2f}")
    print(f"三元组数量: {len(result['triples'])}")
    print(f"冲突数量: {len(result['conflicts'])}")
    print(f"数据源统计: {result['source_stats']}")
    print(f"\n融合上下文:\n{result['fused_context'][:500]}...")
