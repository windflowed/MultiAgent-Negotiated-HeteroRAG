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

        # 构建实体映射表
        self._entity_mapping = self._build_entity_mapping()

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

    def _build_entity_mapping(self) -> Dict[str, str]:
        """
        从知识图谱节点中构建中文名→英文名的映射表

        Returns:
            中文名→英文名的映射字典
        """
        mapping = {}

        if self.kg.graph.number_of_nodes() == 0:
            return mapping

        # 为演员和导演节点构建映射
        for node, attrs in self.kg.graph.nodes(data=True):
            node_type = attrs.get("type", "")
            if node_type in ["actor", "director"]:
                # node 本身是英文名，需要找对应的中文名
                # 通过查询关联的电影来推断，或者使用简单的音译映射
                # 这里先建立英文名→英文名的自映射，后续通过查询补充
                mapping[node.lower()] = node

        return mapping

    def _get_entity_list_for_prompt(self) -> str:
        """
        获取实体列表供 Prompt 使用（限制长度避免 token 过多）

        Returns:
            格式化的实体列表字符串
        """
        if not self.kg.graph.number_of_nodes():
            return "（暂无可用实体列表）"

        # 收集演员和导演节点
        entities = []
        for node, attrs in self.kg.graph.nodes(data=True):
            node_type = attrs.get("type", "")
            if node_type in ["actor", "director"]:
                entities.append(node)

            # 限制列表长度
            if len(entities) >= 100:
                break

        return "\n".join(entities) if entities else "（暂无可用实体列表）"

    def _extract_entities_with_llm(self, query: str) -> List[Dict[str, str]]:
        """
        使用 LLM 从查询中提取实体

        Args:
            query: 用户查询

        Returns:
            实体列表 [{"name": "实体名", "name_en": "英文名", "type": "实体类型"}]
        """
        entity_list = self._get_entity_list_for_prompt()
        prompt = KG_ENTITY_LINK_PROMPT.format(query=query, entity_list=entity_list)

        try:
            response = self.llm.invoke(prompt)
            content = response.content

            # 提取 JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                raw_entities = result.get("entities", [])

                # 进行实体消歧，将中文名转换为 KG 中的英文名
                return self._disambiguate_entities(raw_entities)
        except Exception as e:
            print(f"LLM 实体提取失败: {e}")

        # 回退：使用简单规则提取
        return self._extract_entities_with_rules(query)

    def _disambiguate_entities(
        self, entities: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        实体消歧：将提取的中文实体名转换为 KG 中的英文节点名

        Args:
            entities: LLM 提取的实体列表

        Returns:
            消歧后的实体列表
        """
        if self.kg.graph.number_of_nodes() == 0:
            return entities

        disambiguated = []

        for entity in entities:
            name = entity.get("name", "")
            name_en = entity.get("name_en", "")
            entity_type = entity.get("type", "")

            # 电影类型：KG 中使用中文名，直接使用 name 查询
            if entity_type == "movie":
                if name in self.kg.graph:
                    disambiguated.append({
                        "name": name,
                        "name_en": name,  # 电影使用中文名
                        "type": entity_type
                    })
                else:
                    disambiguated.append(entity)
                continue

            # 对于演员和导演，优先使用英文名
            if entity_type in ["actor", "director"] and name_en:
                # 检查英文名是否在 KG 中
                if name_en in self.kg.graph:
                    disambiguated.append({
                        "name": name,
                        "name_en": name_en,
                        "type": entity_type
                    })
                    continue

            # 尝试直接使用英文名匹配
            if name_en and name_en in self.kg.graph:
                disambiguated.append({
                    "name": name,
                    "name_en": name_en,
                    "type": entity_type
                })
                continue

            # 尝试大小写不敏感匹配
            if name_en:
                name_en_lower = name_en.lower()
                for node in self.kg.graph.nodes():
                    if node.lower() == name_en_lower:
                        disambiguated.append({
                            "name": name,
                            "name_en": node,  # 使用 KG 中的实际名称
                            "type": entity_type
                        })
                        break
                else:
                    # 未找到匹配，保留原始信息
                    disambiguated.append(entity)
            else:
                # 无英文名，尝试直接匹配
                if name in self.kg.graph:
                    disambiguated.append(entity)
                else:
                    # 保留原始信息
                    disambiguated.append(entity)

        return disambiguated

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
                    entities.append({"name": node, "name_en": node, "type": node_type})

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
            # 1. 实体链接（包含消歧）
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

            # 3. 执行查询 - 优先使用 name_en 查询 KG
            if query_type == "common_neighbors" and len(entities) >= 2:
                # 共同邻居查询 - 使用英文名
                entity1 = entities[0].get("name_en") or entities[0]["name"]
                entity2 = entities[1].get("name_en") or entities[1]["name"]
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
                # 两跳查询 - 使用英文名
                entity_name = entities[0].get("name_en") or entities[0]["name"]
                paths = self.kg.query_two_hop(entity_name)

                for path in paths[:10]:  # 限制返回数量
                    result["triples"].append(self._format_path(path))

            else:
                # 单跳查询 - 使用英文名
                for entity in entities[:3]:  # 限制实体数量
                    entity_name = entity.get("name_en") or entity["name"]
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
