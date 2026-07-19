"""简化版测试 - 只处理10条"""
import os, json, time, requests, pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path("D:/pythonproj/MultiAgent-Negotiated-HeteroRAG/.env"))
api_key = os.getenv("TMDB_API_KEY", "")

print("读取数据...")
movies = pd.read_csv("D:/pythonproj/MultiAgent-Negotiated-HeteroRAG/data/raw/tmdb_5000_movies.csv", nrows=10)
print(f"读取 {len(movies)} 条")

print("\n开始请求API...")
for i, (_, row) in enumerate(movies.iterrows()):
    mid = row["id"]
    url = f"https://api.themoviedb.org/3/movie/{mid}"
    resp = requests.get(url, params={"api_key": api_key, "language": "zh-CN"}, timeout=10)
    
    if resp.status_code == 200:
        d = resp.json()
        print(f"[{i+1}] {mid} - {d.get('title', 'N/A')}")
    else:
        print(f"[{i+1}] {mid} - 错误 {resp.status_code}")
    
    time.sleep(0.3)

print("\n完成!")
