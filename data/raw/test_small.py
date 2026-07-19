"""小规模测试 TMDB API"""
import os
import time
import json
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path("D:/pythonproj/MultiAgent-Negotiated-HeteroRAG/.env"))
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")

print(f"API Key: {TMDB_API_KEY[:10]}...")
print(f"API Key 长度: {len(TMDB_API_KEY)}")

# 读取前10条数据
movies_df = pd.read_csv("D:/pythonproj/MultiAgent-Negotiated-HeteroRAG/data/raw/tmdb_5000_movies.csv", nrows=10)
print(f"\n测试数据量: {len(movies_df)} 条")
print(f"电影ID列表: {list(movies_df['id'].values)}")

# 测试API调用
results = []
for idx, row in movies_df.iterrows():
    movie_id = row["id"]
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {"api_key": TMDB_API_KEY, "language": "zh-CN"}
    
    start = time.time()
    try:
        resp = requests.get(url, params=params, timeout=10)
        elapsed = time.time() - start
        
        if resp.status_code == 200:
            data = resp.json()
            title_zh = data.get("title", "")
            overview_zh = data.get("overview", "")[:50]
            results.append({
                "id": movie_id,
                "title": row["title"],
                "title_zh": title_zh,
                "overview_zh": overview_zh,
                "status": "OK",
                "time": f"{elapsed:.2f}s"
            })
            print(f"[{idx+1}] {movie_id} - {title_zh} ({elapsed:.2f}s)")
        else:
            results.append({"id": movie_id, "status": f"ERROR {resp.status_code}", "time": f"{elapsed:.2f}s"})
            print(f"[{idx+1}] {movie_id} - 错误 {resp.status_code} ({elapsed:.2f}s)")
    except Exception as e:
        results.append({"id": movie_id, "status": f"EXCEPTION: {e}", "time": "N/A"})
        print(f"[{idx+1}] {movie_id} - 异常: {e}")
    
    time.sleep(0.05)  # 0.05秒间隔

# 汇总
print("\n" + "="*60)
print("测试汇总")
print("="*60)
ok_count = sum(1 for r in results if r["status"] == "OK")
print(f"成功: {ok_count}/{len(results)}")
print(f"失败: {len(results) - ok_count}/{len(results)}")
