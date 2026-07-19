"""
通用工具函数
包含配置加载、数据库连接等基础功能
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from typing import List, Dict, Any, Optional


def load_config():
    """
    加载项目配置，从环境变量读取所有配置项

    Returns:
        dict: 包含所有配置项的字典
    """
    # 加载 .env 文件
    load_dotenv()

    config = {
        # LLM 配置
        "LLM_API_KEY": os.getenv("LLM_API_KEY", ""),
        "LLM_API_BASE": os.getenv("LLM_API_BASE", "https://api.deepseek.com"),
        "LLM_MODEL_NAME": os.getenv("LLM_MODEL_NAME", "deepseek-v4-flash"),
        # TMDB API 配置
        "TMDB_API_KEY": os.getenv("TMDB_API_KEY", ""),
        # 数据目录配置
        "DATA_DIR": os.getenv("DATA_DIR", "./data"),
        "CHROMA_DB_DIR": os.getenv("CHROMA_DB_DIR", "./chroma_db"),
        "SQLITE_DB_PATH": os.getenv("SQLITE_DB_PATH", "./data/sql/movies.db"),
        "MODEL_DIR": os.getenv("MODEL_DIR", "./models"),
        # Wikipedia 配置
        "WIKI_LANGUAGE": os.getenv("WIKI_LANGUAGE", "zh"),
    }
    return config


def get_project_root():
    """
    获取项目根目录路径

    Returns:
        Path: 项目根目录
    """
    return Path(__file__).parent.parent.parent


def get_data_dir():
    """
    获取数据目录路径

    Returns:
        Path: 数据目录
    """
    config = load_config()
    return Path(config["DATA_DIR"])


def get_sqlite_path():
    """
    获取 SQLite 数据库文件路径

    Returns:
        Path: SQLite 数据库路径
    """
    config = load_config()
    return Path(config["SQLITE_DB_PATH"])


def get_chroma_db_dir():
    """
    获取 ChromaDB 向量库存储路径

    Returns:
        Path: ChromaDB 目录
    """
    config = load_config()
    return Path(config["CHROMA_DB_DIR"])


def get_model_dir():
    """
    获取嵌入模型存储路径

    Returns:
        Path: 模型目录
    """
    config = load_config()
    return Path(config["MODEL_DIR"])


class SQLiteDatabase:
    """SQLite 数据库连接与查询封装"""

    def __init__(self, db_path: Optional[str] = None):
        """
        初始化 SQLite 数据库连接

        Args:
            db_path: 数据库文件路径，为 None 时从配置读取
        """
        if db_path is None:
            config = load_config()
            db_path = config["SQLITE_DB_PATH"]
        self.db_path = Path(db_path)
        self.engine: Engine = create_engine(f"sqlite:///{self.db_path}")

    def get_tables(self) -> List[str]:
        """
        获取数据库所有表名

        Returns:
            表名列表
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ))
            return [row[0] for row in result.fetchall()]

    def get_table_schema(self, table_name: str) -> str:
        """
        获取指定表的建表语句

        Args:
            table_name: 表名

        Returns:
            建表 SQL 语句
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=:name"
            ), {"name": table_name})
            row = result.fetchone()
            return row[0] if row else ""

    def get_all_schemas(self) -> Dict[str, str]:
        """
        获取所有表的建表语句

        Returns:
            {表名: 建表语句} 字典
        """
        schemas = {}
        for table in self.get_tables():
            schemas[table] = self.get_table_schema(table)
        return schemas

    def execute_query(self, sql: str, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        执行 SQL 查询并返回结构化结果

        Args:
            sql: SQL 查询语句
            params: 查询参数

        Returns:
            字典列表，每行一个字典
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            columns = result.keys()
            return [dict(zip(columns, row)) for row in result.fetchall()]

    def execute_query_raw(self, sql: str, params: Optional[Dict] = None) -> List[tuple]:
        """
        执行 SQL 查询并返回原始元组结果

        Args:
            sql: SQL 查询语句
            params: 查询参数

        Returns:
            元组列表
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            return result.fetchall()

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        获取表的详细信息（列名、列类型、行数、样例数据）

        Args:
            table_name: 表名

        Returns:
            包含列信息、行数、样例数据的字典
        """
        with self.engine.connect() as conn:
            # 获取列信息
            result = conn.execute(text(f"PRAGMA table_info({table_name})"))
            columns = [{"name": row[1], "type": row[2], "notnull": bool(row[3])}
                       for row in result.fetchall()]

            # 获取行数
            count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            row_count = count_result.fetchone()[0]

            # 获取样例数据（前3行）
            sample_result = conn.execute(text(f"SELECT * FROM {table_name} LIMIT 3"))
            sample_rows = [dict(zip([c["name"] for c in columns], row))
                           for row in sample_result.fetchall()]

            return {
                "table_name": table_name,
                "columns": columns,
                "row_count": row_count,
                "sample_data": sample_rows
            }


if __name__ == "__main__":
    config = load_config()
    print("项目配置：")
    for key, value in config.items():
        if "KEY" in key:
            print(f"  {key}: {'***' if value else '(未设置)'}")
        else:
            print(f"  {key}: {value}")

    print("\n数据库连接测试：")
    try:
        db = SQLiteDatabase()
        tables = db.get_tables()
        print(f"  表列表: {tables}")
        for table in tables:
            info = db.get_table_info(table)
            print(f"  {table}: {info['row_count']} 行, {len(info['columns'])} 列")
    except Exception as e:
        print(f"  连接失败: {e}")
