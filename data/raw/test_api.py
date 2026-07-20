"""测试 TMDB API 连接"""
import os
import sys
import time
import json
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path("D:/pythonproj/MultiAgent-Negotiated-HeteroRAG/.env"))
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
print(f"API Key: {TMDB_API_KEY[:10]}...")

# Test single request
url = "https://api.themoviedb.org/3/movie/19995"
params = {"api_key": TMDB_API_KEY, "language": "zh-CN"}
try:
    resp = requests.get(url, params=params, timeout=10)
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        title = data.get("title", "")
        print(f"Title: {title}")
except Exception as e:
    print(f"Error: {e}")

# Test reading CSV
movies_file = Path("D:/pythonproj/MultiAgent-Negotiated-HeteroRAG/data/raw/tmdb_5000_movies.csv")
credits_file = Path("D:/pythonproj/MultiAgent-Negotiated-HeteroRAG/data/raw/tmdb_5000_credits.csv")

if movies_file.exists():
    df = pd.read_csv(movies_file, nrows=3)
    print(f"\nMovies CSV columns: {list(df.columns)}")
    print(f"Movies count (full): {len(pd.read_csv(movies_file))}")
else:
    print(f"Movies file not found: {movies_file}")

if credits_file.exists():
    df = pd.read_csv(credits_file, nrows=3)
    print(f"\nCredits CSV columns: {list(df.columns)}")
    print(f"Credits count (full): {len(pd.read_csv(credits_file))}")
else:
    print(f"Credits file not found: {credits_file}")
