"""检查失败的电影"""
import pandas as pd

df = pd.read_csv("D:/pythonproj/MultiAgent-Negotiated-HeteroRAG/data/raw/tmdb_movies.csv")
failed = df[df["title_zh"].isna() | (df["title_zh"] == "")]

print(f"总记录数: {len(df)}")
print(f"失败数量: {len(failed)}")
print(f"\n失败电影详情:")
for _, row in failed.iterrows():
    print(f"  ID: {row['id']}, 英文名: {row['title']}")
