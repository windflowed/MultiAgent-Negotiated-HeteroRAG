"""
通用工具函数
包含配置加载、数据库连接等基础功能
"""

import os
from pathlib import Path
from dotenv import load_dotenv


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


if __name__ == "__main__":
    config = load_config()
    print("项目配置：")
    for key, value in config.items():
        if "KEY" in key:
            print(f"  {key}: {'***' if value else '(未设置)'}")
        else:
            print(f"  {key}: {value}")
