"""验证 API 返回的中文内容"""
import os
import json
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path("D:/pythonproj/MultiAgent-Negotiated-HeteroRAG/.env"))
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")

# 测试3部电影
test_ids = [19995, 285, 559]  # 阿凡达、加勒比海盗、蜘蛛侠
results = []

for movie_id in test_ids:
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {"api_key": TMDB_API_KEY, "language": "zh-CN"}
    resp = requests.get(url, params=params, timeout=10)
    if resp.status_code == 200:
        data = resp.json()
        results.append({
            "id": movie_id,
            "title_en": data.get("title", ""),
            "title_zh": data.get("title", ""),
            "overview_zh": data.get("overview", "")[:100]
        })

# 写入文件查看
output_file = Path("D:/pythonproj/MultiAgent-Negotiated-HeteroRAG/data/raw/test_results.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"结果已写入: {output_file}")
print(f"共 {len(results)} 条记录")
