"""
SQL 查询 Agent
基于 Prompt 模板实现自然语言转 SQL 查询，支持 CoT 思维链与错误重试机制
"""

import json
import re
from typing import Dict, List, Any, Optional

from src.utils.common import load_config, SQLiteDatabase
from src.utils.prompts import SQL_COT_PROMPT, SQL_ERROR_FIX_PROMPT


class SQLQueryAgent:
    """SQL 查询 Agent，支持自然语言转 SQL 查询、CoT 思维链、错误重试"""

    def __init__(self, llm=None):
        """
        初始化 SQL 查询 Agent

        Args:
            llm: 语言模型实例，为 None 时从配置创建
        """
        self.config = load_config()
        self.db = SQLiteDatabase()

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

    def _get_table_schema(self) -> str:
        """获取数据库表结构信息"""
        schemas = self.db.get_all_schemas()
        schema_text = ""
        for table_name, schema in schemas.items():
            info = self.db.get_table_info(table_name)
            schema_text += f"\n表: {table_name}\n"
            schema_text += f"建表语句: {schema}\n"
            schema_text += f"行数: {info['row_count']}\n"
        return schema_text

    def _extract_sql_from_response(self, content: str) -> str:
        """从 LLM 响应中提取 SQL 语句"""
        # 尝试提取代码块中的 SQL
        sql_match = re.search(r'```(?:sql)?\s*(.*?)```', content, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return sql_match.group(1).strip()

        # 尝试提取 SELECT 语句
        select_match = re.search(r'(SELECT\s+.*?)(?:\n|$)', content, re.IGNORECASE | re.DOTALL)
        if select_match:
            return select_match.group(1).strip()

        # 返回最后一行非空内容
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        return lines[-1] if lines else ""

    def _generate_sql_with_cot(self, query: str) -> Dict[str, Any]:
        """
        使用 CoT 思维链生成 SQL 查询

        Args:
            query: 自然语言查询

        Returns:
            包含 SQL 和思考过程的字典
        """
        table_schema = self._get_table_schema()
        prompt = SQL_COT_PROMPT.format(
            table_schema=table_schema,
            query=query
        )

        response = self.llm.invoke(prompt)
        content = response.content

        # 分离思考过程和 SQL
        thinking = content
        sql = self._extract_sql_from_response(content)

        return {
            "thinking": thinking,
            "sql": sql
        }

    def _fix_sql_with_error(self, query: str, sql: str, error: str) -> str:
        """
        根据错误信息修正 SQL

        Args:
            query: 原始自然语言查询
            sql: 出错的 SQL
            error: 错误信息

        Returns:
            修正后的 SQL
        """
        table_schema = self._get_table_schema()
        prompt = SQL_ERROR_FIX_PROMPT.format(
            query=query,
            sql=sql,
            error=error,
            table_schema=table_schema
        )

        response = self.llm.invoke(prompt)
        fixed_sql = self._extract_sql_from_response(response.content)
        return fixed_sql

    def query(self, natural_language_query: str, max_retries: int = 2) -> Dict[str, Any]:
        """
        执行自然语言查询

        Args:
            natural_language_query: 自然语言查询
            max_retries: 最大重试次数（默认2次）

        Returns:
            包含数据、置信度、溯源信息的字典
        """
        result = {
            "success": False,
            "data": None,
            "sql": None,
            "confidence": 0.0,
            "source": "sql",
            "thinking": "",
            "error": None,
            "retries": 0
        }

        # Step 1: 使用 CoT 生成 SQL
        try:
            cot_result = self._generate_sql_with_cot(natural_language_query)
            sql = cot_result["sql"]
            result["thinking"] = cot_result["thinking"]
            result["sql"] = sql
        except Exception as e:
            result["error"] = f"SQL 生成失败: {str(e)}"
            return result

        # Step 2: 执行 SQL 并重试
        for attempt in range(max_retries + 1):
            try:
                query_result = self.db.execute_query(sql)
                result["success"] = True
                result["data"] = query_result
                result["confidence"] = 0.9 if attempt == 0 else 0.8 - (attempt * 0.1)
                result["retries"] = attempt
                return result
            except Exception as e:
                error_msg = str(e)
                if attempt < max_retries:
                    # 尝试修正 SQL
                    try:
                        sql = self._fix_sql_with_error(
                            natural_language_query, sql, error_msg
                        )
                        result["sql"] = sql
                        result["retries"] = attempt + 1
                    except Exception as fix_error:
                        result["error"] = f"SQL 修正失败: {str(fix_error)}"
                        return result
                else:
                    result["error"] = f"查询执行失败（已重试{max_retries}次）: {error_msg}"
                    return result

        return result


def create_sql_agent_instance() -> SQLQueryAgent:
    """创建 SQL 查询 Agent 实例"""
    return SQLQueryAgent()


if __name__ == "__main__":
    print("=" * 60)
    print("SQL 查询 Agent 测试")
    print("=" * 60)

    agent = create_sql_agent_instance()

    # 测试查询
    test_queries = [
        "评分最高的5部电影是什么？",
        "2010年上映的电影有哪些？",
        "票房收入超过10亿的电影有哪些？"
    ]

    for query in test_queries:
        print(f"\n查询: {query}")
        print("-" * 40)
        result = agent.query(query)
        print(f"成功: {result['success']}")
        print(f"SQL: {result['sql']}")
        print(f"置信度: {result['confidence']}")
        print(f"重试次数: {result['retries']}")
        if result['data']:
            print(f"结果数量: {len(result['data'])}")
            for i, row in enumerate(result['data'][:3]):
                print(f"  {i+1}. {row}")
        if result['error']:
            print(f"错误: {result['error']}")
