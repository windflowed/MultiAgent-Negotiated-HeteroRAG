"""TMDB 数据集预处理（并发版）"""
import os, sys, json, time, requests, pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

load_dotenv(Path(__file__).parent.parent.parent / ".env")
api_key = os.getenv("TMDB_API_KEY", "")
base_url = "https://api.themoviedb.org/3"
data_dir = Path(__file__).parent
output_file = data_dir / "tmdb_movies.csv"
log_file = data_dir / "progress.log"

lock = threading.Lock()
stats = {"ok": 0, "fail": 0, "total": 0}


def log(msg):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")


def get_chinese(mid):
    """获取单部电影中文信息"""
    for _ in range(3):
        try:
            r = requests.get(f"{base_url}/movie/{mid}",
                           params={"api_key": api_key, "language": "zh-CN"}, timeout=15)
            if r.status_code == 200:
                d = r.json()
                with lock:
                    stats["ok"] += 1
                return {"title_zh": d.get("title", ""), "overview_zh": d.get("overview", "")}
            elif r.status_code == 429:
                time.sleep(2)
                continue
        except:
            time.sleep(1)
    with lock:
        stats["fail"] += 1
    return {"title_zh": "", "overview_zh": ""}


def parse_json(val):
    if pd.isna(val) or val == "": return []
    try: return json.loads(val) if isinstance(val, str) else val
    except: return []


def main():
    if not api_key:
        log("错误：未配置 TMDB_API_KEY")
        sys.exit(1)

    log("=" * 50)
    log("TMDB 预处理开始（10线程并发）")

    # 读取数据
    movies = pd.read_csv(data_dir / "tmdb_5000_movies.csv")
    credits = pd.read_csv(data_dir / "tmdb_5000_credits.csv")
    credits.columns = ["movie_id", "title_credits", "cast", "crew"]
    merged = pd.merge(movies, credits, left_on="id", right_on="movie_id", how="left")

    total = len(merged)
    stats["total"] = total
    log(f"共 {total} 部电影")

    # 收集所有任务
    tasks = []
    for _, row in merged.iterrows():
        tasks.append({
            "id": row["id"],
            "title": row["title"],
            "original_title": row["original_title"],
            "vote_average": row["vote_average"],
            "revenue": row["revenue"],
            "runtime": row["runtime"],
            "release_date": row["release_date"],
            "genres": row["genres"],
            "cast": row["cast"],
            "crew": row["crew"],
        })

    # 并发请求
    results = {}
    start = time.time()

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_idx = {}
        for i, task in enumerate(tasks):
            future = executor.submit(get_chinese, task["id"])
            future_to_idx[future] = i

        done_count = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            result = future.result()
            results[idx] = result
            done_count += 1

            if done_count % 500 == 0:
                elapsed = time.time() - start
                eta = elapsed / done_count * (total - done_count)
                log(f"进度: {done_count}/{total} (成功:{stats['ok']} 失败:{stats['fail']}) ETA:{eta/60:.0f}分钟")

    log(f"API请求完成! 成功:{stats['ok']} 失败:{stats['fail']}")

    # 格式化输出
    log("生成CSV...")
    rows = []
    for i, task in enumerate(tasks):
        r = results.get(i, {"title_zh": "", "overview_zh": ""})
        rows.append({
            "id": task["id"],
            "title": task["title"],
            "original_title": task["original_title"],
            "title_zh": r["title_zh"],
            "overview_zh": r["overview_zh"],
            "vote_average": task["vote_average"],
            "revenue": task["revenue"],
            "runtime": task["runtime"],
            "release_date": task["release_date"],
            "genres": json.dumps([g["name"] for g in parse_json(task["genres"]) if isinstance(g, dict)], ensure_ascii=False),
            "cast": json.dumps([{"id": c.get("id"), "name": c.get("name"), "character": c.get("character")} for c in parse_json(task["cast"])[:10] if isinstance(c, dict)], ensure_ascii=False),
            "crew": json.dumps([{"id": c.get("id"), "name": c.get("name")} for c in parse_json(task["crew"]) if isinstance(c, dict) and c.get("job") == "Director"], ensure_ascii=False),
        })

    pd.DataFrame(rows).to_csv(output_file, index=False, encoding="utf-8")

    elapsed = time.time() - start
    log(f"完成! 总耗时:{elapsed/60:.1f}分钟")
    log(f"输出: {output_file}")


if __name__ == "__main__":
    main()
