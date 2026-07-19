"""
查询路由 Agent
基于 LLM 零样本分类实现意图识别，输出需要激活的数据源列表
"""

import json
import re
from typing import Dict, List, Any, Optional

from src.utils.common import load_config
from src.utils.prompts import ROUTER_PROMPT


class RouterAgent:
    """查询路由 Agent，基于 LLM 零样本分类实现意图识别"""

    def __init__(self, llm=None, confidence_threshold: float = 0.6):
        """
        初始化查询路由 Agent

        Args:
            llm: 语言模型实例，为 None 时从配置创建
            confidence_threshold: 置信度阈值，低于此值时激活全部数据源
        """
        self.config = load_config()
        self.confidence_threshold = confidence_threshold

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

    def _parse_llm_response(self, content: str) -> Dict[str, Any]:
        """
        解析 LLM 响应

        Args:
            content: LLM 响应内容

        Returns:
            解析后的字典
        """
        # 尝试提取 JSON
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
                return result
            except json.JSONDecodeError:
                pass

        # 回退：使用规则提取
        return self._extract_sources_with_rules(content)

    def _extract_sources_with_rules(self, content: str) -> Dict[str, Any]:
        """
        使用规则提取数据源

        Args:
            content: 响应内容

        Returns:
            提取的数据源字典
        """
        sources = []
        content_lower = content.lower()

        if "sql" in content_lower or "数据库" in content:
            sources.append("sql")
        if "doc" in content_lower or "文档" in content:
            sources.append("doc")
        if "kg" in content_lower or "知识图谱" in content:
            sources.append("kg")

        if not sources:
            sources = ["sql", "doc", "kg"]

        return {"sources": sources, "reason": "规则提取"}

    def _calculate_confidence(
        self,
        sources: List[str],
        query: str
    ) -> float:
        """
        计算路由置信度

        Args:
            sources: 选择的数据源列表
            query: 原始查询

        Returns:
            置信度分数 0.0-1.0
        """
        # 基于数据源数量和查询长度计算置信度
        source_count = len(sources)
        query_length = len(query)

        # 如果选择全部数据源，置信度较低
        if source_count == 3:
            base_confidence = 0.5
        elif source_count == 2:
            base_confidence = 0.7
        else:
            base_confidence = 0.8

        # 查询越长，置信度越高
        length_bonus = min(0.2, query_length / 100)

        return min(1.0, base_confidence + length_bonus)

    def route(self, query: str) -> Dict[str, Any]:
        """
        执行查询路由

        Args:
            query: 自然语言查询

        Returns:
            包含数据源列表、置信度、原因的字典
        """
        result = {
            "success": False,
            "sources": ["sql", "doc", "kg"],  # 默认全部激活
            "confidence": 0.0,
            "reason": "",
            "query": query,
            "error": None
        }

        try:
            # 1. 使用 LLM 进行路由
            prompt = ROUTER_PROMPT.format(query=query)
            response = self.llm.invoke(prompt)
            content = response.content

            # 2. 解析响应
            parsed = self._parse_llm_response(content)
            sources = parsed.get("sources", ["sql", "doc", "kg"])
            reason = parsed.get("reason", "")

            # 3. 验证数据源有效性
            valid_sources = {"sql", "doc", "kg"}
            sources = [s for s in sources if s in valid_sources]

            if not sources:
                sources = ["sql", "doc", "kg"]

            # 4. 计算置信度
            confidence = self._calculate_confidence(sources, query)

            # 5. 如果置信度低于阈值，激活全部数据源
            if confidence < self.confidence_threshold:
                sources = ["sql", "doc", "kg"]
                reason = f"置信度 {confidence:.2f} 低于阈值 {self.confidence_threshold}，默认激活全部数据源"

            result["success"] = True
            result["sources"] = sources
            result["confidence"] = confidence
            result["reason"] = reason

        except Exception as e:
            result["error"] = str(e)

        return result


def create_router_agent(confidence_threshold: float = 0.6) -> RouterAgent:
    """创建查询路由 Agent 实例"""
    return RouterAgent(confidence_threshold=confidence_threshold)


if __name__ == "__main__":
    print("=" * 60)
    print("查询路由 Agent 测试")
    print("=" * 60)

    agent = create_router_agent()

    # 测试查询
    test_queries = [
        "评分最高的5部电影是什么？",  # 应该激活 sql
        "阿凡达的剧情介绍",  # 应该激活 doc
        "詹姆斯·卡梅隆导演了哪些电影？",  # 应该激活 kg
        "阿凡达电影的详细信息",  # 可能激活多个
    ]

    for query in test_queries:
        print(f"\n查询: {query}")
        print("-" * 40)
        result = agent.route(query)
        print(f"成功: {result['success']}")
        print(f"激活数据源: {result['sources']}")
        print(f"置信度: {result['confidence']:.2f}")
        print(f"原因: {result['reason']}")
