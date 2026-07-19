"""
项目配置示例文件
实际配置通过 .env 环境变量读取，此文件仅作为配置项参考
"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ===== LLM 配置 =====
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_API_BASE = os.getenv("LLM_API_BASE", "https://api.deepseek.com")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "deepseek-v4-flash")

# ===== TMDB API 配置 =====
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")

# ===== 数据目录配置 =====
DATA_DIR = os.getenv("DATA_DIR", "./data")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "./data/sql/movies.db")
MODEL_DIR = os.getenv("MODEL_DIR", "./models")

# ===== Wikipedia 配置 =====
WIKI_LANGUAGE = os.getenv("WIKI_LANGUAGE", "zh")
