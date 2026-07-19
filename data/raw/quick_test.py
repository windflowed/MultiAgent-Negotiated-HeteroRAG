"""小规模测试"""
import os, time, json, requests, pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path("D:/pythonproj/MultiAgent-Negotiated-HeteroRAG/.env"))
api_key = os.getenv("TMDB_API_KEY", "")

# 1. 测试API - 只请求3部电影
print("=== 测试1: API连通性 ===")
test_ids = [19995, 285, 559]
for mid in test_ids:
    url = f"https://api.themoviedb.org/3/movie/{mid}"
    resp = requests.get(url, params={"api_key": api_key, "language": "zh-CN"}, timeout=10)
    if resp.status_code == 200:
        d = resp.json()
        with open("D:/pythonproj/MultiAgent-Negotiated-HeteroRAG/data/raw/api_test.txt", "w", encoding="utf-8") as f:
            f.write(f"ID: {mid}\n")
            f.write(f"Title: {d.get('title')}\n")
            f.write(f"Overview: {d.get('overview')[:100]}\n\n")
        print(f"  {mid} OK - {d.get('title')}")
    else:
        print(f"  {mid} FAIL - {resp.status_code}")
    time.sleep(0.5)

# 2. 测试CSV数据
print("\n=== 测试2: CSV数据 ===")
movies = pd.read_csv("D:/pythonproj/MultiAgent-Negotiated-HeteroRAG/data/raw/tmdb_5000_movies.csv")
credits = pd.read_csv("D:/pythonproj/MultiAgent-Negotiated-HeteroRAG/data/raw/tmdb_5000_credits.csv")
print(f"  movies: {len(movies)} rows, columns: {list(movies.columns[:5])}...")
print(f"  credits: {len(credits)} rows, columns: {list(credits.columns)}")
print(f"  movies ID range: {movies['id'].min()} ~ {movies['id'].max()}")
print(f"  有空值的列: {list(movies.columns[movies.isnull().any()])}")

print("\n=== 测试3: 模拟批量请求(5条) ===")
start = time.time()
for mid in movies['id'].head(5):
    resp = requests.get(f"https://api.themoviedb.org/3/movie/{mid}", 
                       params={"api_key": api_key, "language": "zh-CN"}, timeout=10)
    status = "OK" if resp.status_code == 200 else f"ERR {resp.status_code}"
    print(f"  {mid}: {status}")
    time.sleep(0.3)
elapsed = time.time() - start
print(f"  5条耗时: {elapsed:.1f}秒, 平均: {elapsed/5:.1f}秒/条")

print("\n=== 结论 ===")
print(f"  预计处理 {len(movies)} 条需要: {len(movies) * elapsed/5 / 60:.0f} 分钟")
print("  测试完成!")
