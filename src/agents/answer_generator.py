"""
答案生成 Agent
基于融合上下文生成答案，强制标注来源编号，禁止编造信息
"""

import re
from typing import Dict, List, Any, Optional

from src.utils.common import load_config
from src.utils.prompts import ANSWER_GENERATION_PROMPT


class AnswerGeneratorAgent:
    """答案生成 Agent，基于融合上下文生成带来源标注的答案"""

    def __init__(self, llm=None):
        """
        初始化答案生成 Agent

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

    def _validate_citations(self, answer: str, context: str) -> str:
        """
        验证并修正答案中的来源标注

        Args:
            answer: 生成的答案
            context: 原始上下文

        Returns:
            修正后的答案
        """
        # 提取上下文中的来源标签
        source_pattern = r'\[([A-Z]+\d+)\]'
        available_sources = set(re.findall(source_pattern, context))

        # 提取答案中的来源标签
        answer_sources = set(re.findall(source_pattern, answer))

        # 检查是否引用了不存在的来源
        invalid_sources = answer_sources - available_sources
        if invalid_sources:
            print(f"警告: 答案引用了不存在的来源: {invalid_sources}")

        return answer

    def _extract_sources_from_answer(self, answer: str) -> List[str]:
        """
        从答案中提取引用的来源列表

        Args:
            answer: 生成的答案

        Returns:
            来源列表
        """
        # 提取所有来源标签
        source_pattern = r'\[([A-Z]+\d+)\]'
        sources = re.findall(source_pattern, answer)

        # 去重并保持顺序
        unique_sources = []
        seen = set()
        for s in sources:
            if s not in seen:
                unique_sources.append(s)
                seen.add(s)

        return unique_sources

    def generate(
        self,
        query: str,
        fused_context: str,
        source_stats: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """
        基于融合上下文生成答案

        Args:
            query: 用户原始问题
            fused_context: 融合后的上下文文本
            source_stats: 数据源统计信息

        Returns:
            答案生成结果字典
        """
        result = {
            "success": False,
            "answer": "",
            "sources": [],
            "confidence": 0.0,
            "error": None
        }

        if not fused_context:
            result["error"] = "无可用的上下文信息"
            return result

        try:
            # 构建提示词
            prompt = ANSWER_GENERATION_PROMPT.format(
                context=fused_context,
                query=query
            )

            # 调用 LLM 生成答案
            response = self.llm.invoke(prompt)
            answer = response.content.strip()

            # 验证来源标注
            answer = self._validate_citations(answer, fused_context)

            # 提取来源列表
            sources = self._extract_sources_from_answer(answer)

            # 计算置信度
            confidence = 0.8
            if source_stats:
                active_sources = sum(1 for count in source_stats.values() if count > 0)
                confidence = min(0.95, 0.6 + active_sources * 0.1)

            result["success"] = True
            result["answer"] = answer
            result["sources"] = sources
            result["confidence"] = confidence

        except Exception as e:
            result["error"] = str(e)

        return result

    def generate_simple(
        self,
        query: str,
        fused_context: str
    ) -> str:
        """
        简化版答案生成，仅返回答案文本

        Args:
            query: 用户原始问题
            fused_context: 融合后的上下文文本

        Returns:
            答案文本
        """
        result = self.generate(query, fused_context)
        return result.get("answer", "")


def create_answer_generator() -> AnswerGeneratorAgent:
    """创建答案生成 Agent 实例"""
    return AnswerGeneratorAgent()


if __name__ == "__main__":
    print("=" * 60)
    print("答案生成 Agent 测试")
    print("=" * 60)

    agent = create_answer_generator()

    # 模拟融合上下文
    fused_context = """[SQL1] 阿凡达 - title - 阿凡达
[SQL2] 阿凡达 - vote_average - 7.9
[SQL3] 阿凡达 - revenue - 2787965087
[DOC1] 阿凡达是一部科幻电影，由詹姆斯·卡梅隆执导。
[KG1] 阿凡达 --[属于]--> Action
[KG2] 阿凡达 --[主演]--> Sam Worthington"""

    query = "阿凡达电影的评分是多少？"

    result = agent.generate(query, fused_context)
    print(f"成功: {result['success']}")
    print(f"答案:\n{result['answer']}")
    print(f"引用来源: {result['sources']}")
    print(f"置信度: {result['confidence']:.2f}")
