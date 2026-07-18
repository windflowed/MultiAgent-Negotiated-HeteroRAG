"""TASK-007: 生成中文 SQLite 数据库"""
import os
import json
import sqlite3
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent
CSV_FILE = DATA_DIR / "tmdb_movies.csv"
DB_FILE = DATA_DIR.parent / "sql" / "movies.db"


def create_tables(conn):
    """创建7张表"""
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS movies (
        id INTEGER PRIMARY KEY,
        title TEXT,
        title_zh TEXT,
        overview_zh TEXT,
        vote_average REAL,
        revenue REAL,
        runtime INTEGER,
        release_date TEXT
    )""")
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS genres (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE
    )""")
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS movie_genres (
        movie_id INTEGER,
        genre_id INTEGER,
        PRIMARY KEY (movie_id, genre_id),
        FOREIGN KEY (movie_id) REFERENCES movies(id),
        FOREIGN KEY (genre_id) REFERENCES genres(id)
    )""")
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS actors (
        id INTEGER PRIMARY KEY,
        name TEXT
    )""")
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS directors (
        id INTEGER PRIMARY KEY,
        name TEXT
    )""")
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS movie_actors (
        movie_id INTEGER,
        actor_id INTEGER,
        PRIMARY KEY (movie_id, actor_id),
        FOREIGN KEY (movie_id) REFERENCES movies(id),
        FOREIGN KEY (actor_id) REFERENCES actors(id)
    )""")
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS movie_directors (
        movie_id INTEGER,
        director_id INTEGER,
        PRIMARY KEY (movie_id, director_id),
        FOREIGN KEY (movie_id) REFERENCES movies(id),
        FOREIGN KEY (director_id) REFERENCES directors(id)
    )""")
    
    conn.commit()


def parse_json(val):
    if pd.isna(val) or val == "":
        return []
    try:
        if isinstance(val, str):
            parsed = json.loads(val)
            return parsed if isinstance(parsed, list) else []
        return val if isinstance(val, list) else []
    except:
        return []


def main():
    print("=" * 50)
    print("TASK-007: 生成中文 SQLite 数据库")
    print("=" * 50)
    
    # 读取CSV
    print("\n[1/4] 读取 CSV...")
    df = pd.read_csv(CSV_FILE)
    print(f"  读取 {len(df)} 条记录")
    
    # 创建数据库目录
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # 删除旧数据库
    if DB_FILE.exists():
        DB_FILE.unlink()
    
    # 连接数据库
    conn = sqlite3.connect(str(DB_FILE))
    
    # 创建表
    print("\n[2/4] 创建表结构...")
    create_tables(conn)
    
    # 插入电影数据
    print("\n[3/4] 插入数据...")
    cursor = conn.cursor()
    
    movies_data = []
    genres_set = set()
    movie_genres_data = []
    actors_set = {}
    directors_set = {}
    movie_actors_data = []
    movie_directors_data = []
    
    for _, row in df.iterrows():
        movie_id = int(row["id"])
        
        # 电影表
        movies_data.append((
            movie_id,
            str(row["title"]) if pd.notna(row["title"]) else "",
            str(row["title_zh"]) if pd.notna(row["title_zh"]) else "",
            str(row["overview_zh"]) if pd.notna(row["overview_zh"]) else "",
            float(row["vote_average"]) if pd.notna(row["vote_average"]) else 0.0,
            float(row["revenue"]) if pd.notna(row["revenue"]) else 0.0,
            int(row["runtime"]) if pd.notna(row["runtime"]) else 0,
            str(row["release_date"]) if pd.notna(row["release_date"]) else ""
        ))
        
        # 类型 - 格式: ["Action", "Adventure"]
        genres_list = parse_json(row["genres"])
        for g_name in genres_list:
            if isinstance(g_name, str) and g_name:
                genres_set.add(g_name)
                movie_genres_data.append((movie_id, g_name))
        
        # 演员 - 格式: [{"id": xxx, "name": "xxx"}]
        cast_list = parse_json(row["cast"])
        for c in cast_list:
            if isinstance(c, dict):
                actor_id = c.get("id", 0)
                actor_name = c.get("name", "")
                if actor_id and actor_name:
                    actors_set[actor_id] = actor_name
                    movie_actors_data.append((movie_id, actor_id))
        
        # 导演 - 格式: [{"id": xxx, "name": "xxx"}]
        crew_list = parse_json(row["crew"])
        for c in crew_list:
            if isinstance(c, dict):
                director_id = c.get("id", 0)
                director_name = c.get("name", "")
                if director_id and director_name:
                    directors_set[director_id] = director_name
                    movie_directors_data.append((movie_id, director_id))
    
    # 批量插入
    cursor.executemany("INSERT OR IGNORE INTO movies VALUES (?,?,?,?,?,?,?,?)", movies_data)
    
    # 类型表
    for g_name in genres_set:
        cursor.execute("INSERT OR IGNORE INTO genres (name) VALUES (?)", (g_name,))
    
    # 类型ID映射
    genre_id_map = {}
    for row in cursor.execute("SELECT id, name FROM genres"):
        genre_id_map[row[1]] = row[0]
    
    movie_genres_final = [(mid, genre_id_map[g_name]) for mid, g_name in movie_genres_data if g_name in genre_id_map]
    cursor.executemany("INSERT OR IGNORE INTO movie_genres VALUES (?,?)", movie_genres_final)
    
    # 演员表
    for actor_id, actor_name in actors_set.items():
        cursor.execute("INSERT OR IGNORE INTO actors (id, name) VALUES (?, ?)", (actor_id, actor_name))
    
    cursor.executemany("INSERT OR IGNORE INTO movie_actors VALUES (?,?)", movie_actors_data)
    
    # 导演表
    for director_id, director_name in directors_set.items():
        cursor.execute("INSERT OR IGNORE INTO directors (id, name) VALUES (?, ?)", (director_id, director_name))
    
    cursor.executemany("INSERT OR IGNORE INTO movie_directors VALUES (?,?)", movie_directors_data)
    
    conn.commit()
    
    # 统计
    print("\n[4/4] 验证结果...")
    stats = {}
    for table in ["movies", "genres", "movie_genres", "actors", "directors", "movie_actors", "movie_directors"]:
        count = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        stats[table] = count
        print(f"  {table}: {count} 条")
    
    conn.close()
    
    print("\n" + "=" * 50)
    print(f"数据库生成完成: {DB_FILE}")
    print(f"文件大小: {DB_FILE.stat().st_size / 1024:.1f} KB")
    print("=" * 50)


if __name__ == "__main__":
    main()
