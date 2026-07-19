"""
知识图谱 Agent
实现实体链接功能，根据查询执行对应跳数的图谱推理，返回三元组结果
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple

from src.utils.common import load_config
from src.utils.kg_utils import KnowledgeGraph
from src.utils.prompts import KG_ENTITY_LINK_PROMPT


class KGAgent:
    """知识图谱 Agent，支持实体链接、图谱推理、三元组查询"""

    def __init__(self, llm=None):
        """
        初始化知识图谱 Agent

        Args:
            llm: 语言模型实例，为 None 时从配置创建
        """
        self.config = load_config()
        self.kg = KnowledgeGraph()

        # 加载知识图谱
        try:
            self.kg.load()
        except Exception:
            print("知识图谱未找到，需要先构建")

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

    def _extract_entities_with_llm(self, query: str) -> List[Dict[str, str]]:
        """
        使用 LLM 从查询中提取实体

        Args:
            query: 用户查询

        Returns:
            实体列表 [{"name": "实体名", "type": "实体类型"}]
        """
        prompt = KG_ENTITY_LINK_PROMPT.format(query=query)

        try:
            response = self.llm.invoke(prompt)
            content = response.content

            # 提取 JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result.get("entities", [])
        except Exception as e:
            print(f"LLM 实体提取失败: {e}")

        # 回退：使用简单规则提取
        return self._extract_entities_with_rules(query)

    def _extract_entities_with_rules(self, query: str) -> List[Dict[str, str]]:
        """
        使用规则从查询中提取实体

        Args:
            query: 用户查询

        Returns:
            实体列表
        """
        entities = []

        # 检查知识图谱中存在的实体
        if self.kg.graph.number_of_nodes() > 0:
            for node in self.kg.graph.nodes():
                if node in query:
                    node_type = self.kg.graph.nodes[node].get("type", "unknown")
                    entities.append({"name": node, "type": node_type})

        return entities

    def _determine_query_type(
        self,
        entities: List[Dict[str, str]],
        query: str
    ) -> str:
        """
        根据实体和查询判断查询类型

        Args:
            entities: 提取的实体列表
            query: 原始查询

        Returns:
            查询类型: "single_hop", "two_hop", "common_neighbors"
        """
        # 如果有两个实体，可能是共同邻居查询
        if len(entities) >= 2:
            # 检查查询中是否包含合作、共同等关键词
            cooperation_keywords = ["合作", "共同", "一起", "搭档"]
            for keyword in cooperation_keywords:
                if keyword in query:
                    return "common_neighbors"

        # 默认使用单跳查询
        return "single_hop"

    def _format_triple(
        self,
        triple: Tuple[str, str, str],
        confidence: float = 0.8
    ) -> Dict[str, Any]:
        """
        格式化三元组

        Args:
            triple: (源实体, 关系, 目标实体)
            confidence: 置信度

        Returns:
            格式化后的三元组字典
        """
        return {
            "subject": triple[0],
            "predicate": triple[1],
            "object": triple[2],
            "confidence": confidence,
            "source": "kg"
        }

    def _format_path(
        self,
        path: Tuple[str, str, str, str, str],
        confidence: float = 0.7
    ) -> Dict[str, Any]:
        """
        格式化路径（两跳）

        Args:
            path: (实体1, 关系1, 中间实体, 关系2, 实体2)
            confidence: 置信度

        Returns:
            格式化后的路径字典
        """
        return {
            "subject": path[0],
            "predicate1": path[1],
            "middle": path[2],
            "predicate2": path[3],
            "object": path[4],
            "confidence": confidence,
            "source": "kg",
            "triples": [
                {"subject": path[0], "predicate": path[1], "object": path[2]},
                {"subject": path[2], "predicate": path[3], "object": path[4]}
            ]
        }

    def query(self, natural_language_query: str) -> Dict[str, Any]:
        """
        执行知识图谱查询

        Args:
            natural_language_query: 自然语言查询

        Returns:
            包含三元组、置信度、溯源信息的字典
        """
        result = {
            "success": False,
            "triples": [],
            "query_type": None,
            "entities_found": [],
            "source": "kg",
            "confidence": 0.0,
            "error": None
        }

        try:
            # 1. 实体链接
            entities = self._extract_entities_with_llm(natural_language_query)
            result["entities_found"] = entities

            if not entities:
                result["error"] = "未识别到相关实体"
                return result

            # 2. 判断查询类型
            query_type = self._determine_query_type(
                entities, natural_language_query
            )
            result["query_type"] = query_type

            # 3. 执行查询
            if query_type == "common_neighbors" and len(entities) >= 2:
                # 共同邻居查询
                entity1 = entities[0]["name"]
                entity2 = entities[1]["name"]
                common = self.kg.query_common_neighbors(entity1, entity2)

                for neighbor in common:
                    result["triples"].append(self._format_triple(
                        (entity1, "共同", neighbor),
                        confidence=0.7
                    ))
                    result["triples"].append(self._format_triple(
                        (entity2, "共同", neighbor),
                        confidence=0.7
                    ))

            elif query_type == "two_hop":
                # 两跳查询
                entity_name = entities[0]["name"]
                paths = self.kg.query_two_hop(entity_name)

                for path in paths[:10]:  # 限制返回数量
                    result["triples"].append(self._format_path(path))

            else:
                # 单跳查询
                for entity in entities[:3]:  # 限制实体数量
                    entity_name = entity["name"]
                    triples = self.kg.query_single_hop(entity_name)

                    for triple in triples[:10]:  # 限制返回数量
                        result["triples"].append(self._format_triple(triple))

            # 4. 计算置信度
            if result["triples"]:
                result["success"] = True
                result["confidence"] = 0.8 if len(result["triples"]) > 0 else 0.5

        except Exception as e:
            result["error"] = str(e)

        return result


def create_kg_agent() -> KGAgent:
    """创建知识图谱 Agent 实例"""
    return KGAgent()


if __name__ == "__main__":
    print("=" * 60)
    print("知识图谱 Agent 测试")
    print("=" * 60)

    agent = create_kg_agent()

    # 测试查询
    test_queries = [
        "阿凡达有哪些演员？",
        "詹姆斯·卡梅隆导演了哪些电影？",
        "阿凡达和泰坦尼克号有什么共同演员？"
    ]

    for query in test_queries:
        print(f"\n查询: {query}")
        print("-" * 40)
        result = agent.query(query)
        print(f"成功: {result['success']}")
        print(f"查询类型: {result['query_type']}")
        print(f"识别到的实体: {result['entities_found']}")
        print(f"三元组数量: {len(result['triples'])}")
        for i, triple in enumerate(result['triples'][:3]):
            print(f"  {i+1}. {triple}")
