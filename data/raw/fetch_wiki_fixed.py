"""TASK-008: 抓取中文 Wikipedia 文档 - 修复版（带持久化失败跳过）"""
import requests
import sqlite3
import time
import logging
import sys
from pathlib import Path

# 路径配置
DB_FILE = Path(r"D:\pythonproj\MultiAgent-Negotiated-HeteroRAG\data\sql\movies.db")
DOCS_DIR = Path(r"D:\pythonproj\MultiAgent-Negotiated-HeteroRAG\data\documents")
LOGS_DIR = Path(r"D:\pythonproj\MultiAgent-Negotiated-HeteroRAG\data\logs")
FAILED_LOG = LOGS_DIR / "wiki_failed.log"
FAILED_TITLES_FILE = LOGS_DIR / "wiki_failed_titles.txt"

# 创建目录
DOCS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# 配置日志
logging.basicConfig(
    filename=str(FAILED_LOG),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# Wikipedia API 配置
WIKI_API_URL = "https://zh.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "MultiAgentRAG/1.0 (https://github.com/MultiAgent-Negotiated-HeteroRAG)"}


def fetch_wiki_content(title_zh, retries=3):
    """获取 Wikipedia 页面内容，带重定向跟随和重试机制"""
    params = {
        "action": "query",
        "titles": title_zh,
        "prop": "extracts",
        "exintro": False,
        "explaintext": True,
        "exlimit": 1,
        "redirects": 1,  # 跟随重定向
        "format": "json"
    }
    
    for attempt in range(retries):
        try:
            r = requests.get(WIKI_API_URL, params=params, headers=HEADERS, timeout=15)
            
            # 处理 429 限速
            if r.status_code == 429:
                wait_time = 2 ** (attempt + 2)  # 4, 8, 16 秒
                print(f"    [429] Rate limited, waiting {wait_time}s...", flush=True)
                time.sleep(wait_time)
                continue
            
            r.raise_for_status()
            data = r.json()
            pages = data.get("query", {}).get("pages", {})
            
            for pid, page in pages.items():
                if pid == "-1":
                    return None, "not_found"
                extract = page.get("extract", "")
                if not extract:
                    return None, "empty"
                return extract, "ok"
            
            return None, "error"
            
        except requests.exceptions.Timeout:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                logging.error(f"Timeout: {title_zh}")
                return None, "timeout"
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
            else:
                logging.error(f"Error: {title_zh}: {e}")
                return None, "error"
    
    return None, "max_retries"


def load_failed_titles():
    """加载已知失败的电影标题集合"""
    if FAILED_TITLES_FILE.exists():
        titles = set(FAILED_TITLES_FILE.read_text(encoding='utf-8').strip().splitlines())
        titles.discard('')
        return titles
    return set()


def save_failed_titles(failed_set):
    """持久化失败标题集合"""
    FAILED_TITLES_FILE.write_text('\n'.join(sorted(failed_set)), encoding='utf-8')


def main():
    """主函数 - 抓取所有电影的 Wikipedia 文档"""
    print("=" * 60)
    print("TASK-008: 抓取中文 Wikipedia 文档")
    print("=" * 60)
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 读取电影列表
    print("[1/3] 读取电影列表...")
    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, title, title_zh, vote_average, revenue, runtime, release_date 
        FROM movies 
        WHERE title_zh IS NOT NULL AND title_zh != ''
    """)
    movies = cursor.fetchall()
    conn.close()
    
    total_movies = len(movies)
    print(f"  共 {total_movies} 部电影有中文标题")
    
    # 加载已知失败标题（避免重复请求）
    known_failed = load_failed_titles()
    print(f"  已知失败标题: {len(known_failed)} 个")
    print()
    
    # 统计
    success = 0
    failed = 0
    skipped = 0
    new_failed = set()
    
    # 开始抓取
    print("[2/3] 开始抓取 Wikipedia 文档...")
    print("-" * 60)
    
    start_time = time.time()
    
    for i, (movie_id, title, title_zh, vote_average, revenue, runtime, release_date) in enumerate(movies):
        # 进度显示
        if i % 10 == 0 or i == total_movies - 1:
            elapsed = time.time() - start_time
            rate = success / (elapsed / 60) if elapsed > 60 else success
            print(f"[{i+1}/{total_movies}] OK:{success} Fail:{failed} Skip:{skipped} | {rate:.0f} docs/min", flush=True)
        
        # 检查文件是否已存在
        safe_title = "".join(c for c in title_zh if c.isalnum() or c in "._- ")
        file_path = DOCS_DIR / f"{safe_title}.md"
        
        if file_path.exists():
            skipped += 1
            continue
        
        # 跳过已知失败的标题
        if title_zh in known_failed:
            skipped += 1
            continue
        
        # 获取 Wikipedia 内容
        content, status = fetch_wiki_content(title_zh)
        
        if status == "not_found":
            logging.warning(f"Not found: {title_zh}")
            new_failed.add(title_zh)
            failed += 1
            if len(new_failed) % 50 == 0:
                known_failed.update(new_failed)
                save_failed_titles(known_failed)
            continue
        
        if status == "empty":
            logging.info(f"Empty content: {title_zh}")
            new_failed.add(title_zh)
            failed += 1
            if len(new_failed) % 50 == 0:
                known_failed.update(new_failed)
                save_failed_titles(known_failed)
            continue
        
        if content is None:
            new_failed.add(title_zh)
            failed += 1
            if len(new_failed) % 50 == 0:
                known_failed.update(new_failed)
                save_failed_titles(known_failed)
            continue
        
        # 检查内容长度（至少50字）
        if len(content) < 50:
            logging.info(f"Short content ({len(content)}): {title_zh}")
            new_failed.add(title_zh)
            skipped += 1
            if len(new_failed) % 50 == 0:
                known_failed.update(new_failed)
                save_failed_titles(known_failed)
            continue
        
        # 保存 Markdown
        md_content = f"""# {title_zh}

## 基本信息

- **电影ID**: {movie_id}
- **英文名**: {title}
- **评分**: {vote_average}
- **票房**: {revenue}
- **片长**: {runtime} 分钟
- **上映日期**: {release_date}

## 维基百科内容

{content}
"""
        file_path.write_text(md_content, encoding='utf-8')
        success += 1
        
        # 顺序请求限速（0.15秒间隔）
        time.sleep(0.15)
    
    # 持久化失败标题
    known_failed.update(new_failed)
    save_failed_titles(known_failed)
    
    # 最终统计
    elapsed = time.time() - start_time
    total_files = len(list(DOCS_DIR.glob("*.md")))
    
    print()
    print("-" * 60)
    print("[3/3] 抓取完成")
    print("=" * 60)
    print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总用时: {elapsed:.0f} 秒 ({elapsed/60:.1f} 分钟)")
    print()
    print("统计结果:")
    print(f"  总电影数: {total_movies}")
    print(f"  成功抓取: {success}")
    print(f"  失败: {failed}")
    print(f"  跳过: {skipped}")
    print(f"  文档目录文件数: {total_files}")
    print(f"  持久化失败标题: {len(known_failed)} 个")
    print()
    print(f"失败日志: {FAILED_LOG}")
    print("=" * 60)


if __name__ == "__main__":
    main()
