"""TASK-008: 抓取中文 Wikipedia 文档"""
import requests
import sqlite3
import time
import logging
from pathlib import Path

# 路径配置
DB_FILE = Path(__file__).parent.parent / "sql" / "movies.db"
DOCS_DIR = Path(__file__).parent / "documents"
LOGS_DIR = Path(__file__).parent / "logs"
FAILED_LOG = LOGS_DIR / "wiki_failed.log"

# 创建目录
DOCS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# 配置失败日志
logging.basicConfig(
    filename=str(FAILED_LOG),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# Wikipedia API 配置
WIKI_API_URL = "https://zh.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "MultiAgentRAG/1.0"}


def fetch_wiki_content(title_zh, retries=2):
    """获取 Wikipedia 页面内容，支持重试"""
    params = {
        "action": "query",
        "titles": title_zh,
        "prop": "extracts",
        "exintro": False,
        "explaintext": True,
        "exlimit": 1,
        "format": "json"
    }
    
    for attempt in range(retries + 1):
        try:
            r = requests.get(WIKI_API_URL, params=params, headers=HEADERS, timeout=10)
            r.raise_for_status()
            data = r.json()
            pages = data.get("query", {}).get("pages", {})
            
            for pid, page in pages.items():
                if pid == "-1":
                    return None
                return page.get("extract", "")
            
            return None
        except Exception as e:
            if attempt < retries:
                time.sleep(1)
            else:
                logging.error(f"Failed to fetch '{title_zh}' after {retries} retries: {e}")
                return None


def main():
    print("=" * 50)
    print("TASK-008: 抓取中文 Wikipedia 文档")
    print("=" * 50)
    
    # 连接数据库
    print("\n[1/4] 读取电影列表...")
    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, title_zh, vote_average, revenue, runtime, release_date FROM movies WHERE title_zh != '' AND title_zh IS NOT NULL")
    movies = cursor.fetchall()
    conn.close()
    
    total_movies = len(movies)
    print(f"  共 {total_movies} 部电影有中文标题")
    
    # 统计
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    # 抓取文档
    print("\n[2/4] 开始抓取 Wikipedia 文档...")
    start_time = time.time()
    
    for i, (movie_id, title, title_zh, vote_average, revenue, runtime, release_date) in enumerate(movies):
        # 进度显示
        if i % 50 == 0:
            elapsed = time.time() - start_time
            print(f"  进度: {i}/{total_movies} ({i*100//total_movies}%) | 成功:{success_count} 失败:{failed_count} 跳过:{skipped_count} | 用时:{elapsed:.0f}s")
        
        # 检查文件是否已存在
        safe_title = "".join(c for c in title_zh if c.isalnum() or c in "._- ")
        file_path = DOCS_DIR / f"{safe_title}.md"
        if file_path.exists():
            skipped_count += 1
            continue
        
        # 获取 Wikipedia 内容
        content = fetch_wiki_content(title_zh)
        
        if content is None:
            logging.warning(f"Page not found: {title_zh}")
            failed_count += 1
            continue
        
        # 检查内容长度
        if len(content) < 100:
            logging.info(f"Content too short ({len(content)} chars): {title_zh}")
            skipped_count += 1
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
        success_count += 1
        
        # 限速
        time.sleep(0.1)
    
    # 统计结果
    print("\n[3/4] 统计结果...")
    total_files = len(list(DOCS_DIR.glob("*.md")))
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 50)
    print("抓取完成")
    print(f"  总电影数: {total_movies}")
    print(f"  成功抓取: {success_count}")
    print(f"  失败: {failed_count}")
    print(f"  跳过(已存在/内容短): {skipped_count}")
    print(f"  文档目录文件总数: {total_files}")
    print(f"  总用时: {elapsed:.0f} 秒")
    print(f"  失败日志: {FAILED_LOG}")
    print("=" * 50)


if __name__ == "__main__":
    main()
