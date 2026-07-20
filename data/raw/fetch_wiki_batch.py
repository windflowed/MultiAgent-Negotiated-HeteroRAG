"""TASK-008: 分批抓取中文 Wikipedia 文档 - 改进版"""
import requests
import sqlite3
import time
import logging
from pathlib import Path

# 路径配置
DB_FILE = Path(r"D:\pythonproj\MultiAgent-Negotiated-HeteroRAG\data\sql\movies.db")
DOCS_DIR = Path(r"D:\pythonproj\MultiAgent-Negotiated-HeteroRAG\data\documents")
LOGS_DIR = Path(r"D:\pythonproj\MultiAgent-Negotiated-HeteroRAG\data\logs")
FAILED_LOG = LOGS_DIR / "wiki_failed.log"

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
HEADERS = {"User-Agent": "MultiAgentRAG/1.0"}


def fetch_wiki_content(title_zh, retries=3):
    """获取 Wikipedia 页面内容，带限速处理"""
    params = {
        "action": "query",
        "titles": title_zh,
        "prop": "extracts",
        "exintro": False,
        "explaintext": True,
        "exlimit": 1,
        "format": "json"
    }
    
    for attempt in range(retries):
        try:
            r = requests.get(WIKI_API_URL, params=params, headers=HEADERS, timeout=15)
            
            # 处理 429 限速
            if r.status_code == 429:
                wait_time = 2 ** (attempt + 1)  # 指数退避
                print(f"  Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            
            r.raise_for_status()
            data = r.json()
            pages = data.get("query", {}).get("pages", {})
            
            for pid, page in pages.items():
                if pid == "-1":
                    return None
                return page.get("extract", "")
            
            return None
            
        except requests.exceptions.Timeout:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                logging.error(f"Timeout: {title_zh}")
                return None
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
            else:
                logging.error(f"Error: {title_zh}: {e}")
                return None
    
    return None


def main():
    """主函数"""
    import sys
    
    start_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    print(f"Batch: {start_idx} to {start_idx + batch_size}")
    
    # 读取电影列表
    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, title_zh, vote_average, revenue, runtime, release_date FROM movies WHERE title_zh != '' AND title_zh IS NOT NULL")
    movies = cursor.fetchall()
    conn.close()
    
    batch = movies[start_idx:start_idx + batch_size]
    
    success = 0
    failed = 0
    skipped = 0
    
    for i, (movie_id, title, title_zh, vote_average, revenue, runtime, release_date) in enumerate(batch):
        if i % 10 == 0:
            print(f"  {i}/{len(batch)} - OK:{success} Fail:{failed} Skip:{skipped}")
        
        safe_title = "".join(c for c in title_zh if c.isalnum() or c in "._- ")
        file_path = DOCS_DIR / f"{safe_title}.md"
        
        if file_path.exists():
            skipped += 1
            continue
        
        content = fetch_wiki_content(title_zh)
        
        if content is None:
            logging.warning(f"Not found: {title_zh}")
            failed += 1
            continue
        
        if len(content) < 100:
            logging.info(f"Short ({len(content)}): {title_zh}")
            skipped += 1
            continue
        
        md = f"""# {title_zh}

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
        file_path.write_text(md, encoding='utf-8')
        success += 1
        
        # 基础限速
        time.sleep(0.2)
    
    print(f"\nDone: {success} success, {failed} failed, {skipped} skipped")
    print(f"Total files: {len(list(DOCS_DIR.glob('*.md')))}")


if __name__ == "__main__":
    main()
