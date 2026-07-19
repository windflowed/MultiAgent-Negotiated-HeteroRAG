"""
NetworkX 知识图谱模块
从 SQLite 数据库抽取实体与关系，构建内存有向图，封装查询接口
"""

import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

import networkx as nx

from src.utils.common import SQLiteDatabase


class KnowledgeGraph:
    """知识图谱封装类"""

    def __init__(self):
        """初始化知识图谱"""
        self.graph = nx.DiGraph()

    def build_graph(self) -> None:
        """
        从 SQLite 数据库构建知识图谱
        """
        db = SQLiteDatabase()

        # 1. 加载电影节点
        movies = db.execute_query("SELECT id, title_zh FROM movies WHERE title_zh IS NOT NULL AND title_zh != ''")
        for movie in movies:
            self.graph.add_node(
                movie["title_zh"],
                type="movie",
                movie_id=movie["id"]
            )

        # 2. 加载类型节点和电影-类型关系
        genres = db.execute_query("SELECT id, name FROM genres")
        for genre in genres:
            self.graph.add_node(genre["name"], type="genre")

        movie_genres = db.execute_query("""
            SELECT m.title_zh, g.name
            FROM movie_genres mg
            JOIN movies m ON mg.movie_id = m.id
            JOIN genres g ON mg.genre_id = g.id
            WHERE m.title_zh IS NOT NULL AND m.title_zh != ''
        """)
        for mg in movie_genres:
            if self.graph.has_node(mg["title_zh"]) and self.graph.has_node(mg["name"]):
                self.graph.add_edge(mg["title_zh"], mg["name"], relation="属于")

        # 3. 加载演员节点和电影-演员关系
        actors = db.execute_query("SELECT id, name FROM actors WHERE name IS NOT NULL AND name != ''")
        for actor in actors:
            self.graph.add_node(actor["name"], type="actor")

        movie_actors = db.execute_query("""
            SELECT m.title_zh, a.name
            FROM movie_actors ma
            JOIN movies m ON ma.movie_id = m.id
            JOIN actors a ON ma.actor_id = a.id
            WHERE m.title_zh IS NOT NULL AND m.title_zh != '' AND a.name IS NOT NULL AND a.name != ''
        """)
        actor_movies = defaultdict(list)
        for ma in movie_actors:
            if self.graph.has_node(ma["title_zh"]) and self.graph.has_node(ma["name"]):
                self.graph.add_edge(ma["title_zh"], ma["name"], relation="主演")
                actor_movies[ma["name"]].append(ma["title_zh"])

        # 4. 加载导演节点和电影-导演关系
        directors = db.execute_query("SELECT id, name FROM directors WHERE name IS NOT NULL AND name != ''")
        for director in directors:
            self.graph.add_node(director["name"], type="director")

        movie_directors = db.execute_query("""
            SELECT m.title_zh, d.name
            FROM movie_directors md
            JOIN movies m ON md.movie_id = m.id
            JOIN directors d ON md.director_id = d.id
            WHERE m.title_zh IS NOT NULL AND m.title_zh != '' AND d.name IS NOT NULL AND d.name != ''
        """)
        for md in movie_directors:
            if self.graph.has_node(md["title_zh"]) and self.graph.has_node(md["name"]):
                self.graph.add_edge(md["title_zh"], md["name"], relation="导演")

        # 5. 构建演员合作关系（共同出演同一部电影）
        for actor, movies_list in actor_movies.items():
            for i in range(len(movies_list)):
                for j in range(i + 1, len(movies_list)):
                    if self.graph.has_node(movies_list[i]) and self.graph.has_node(movies_list[j]):
                        if not self.graph.has_edge(actor, movies_list[i]):
                            self.graph.add_edge(actor, movies_list[i], relation="合作出演")
                        if not self.graph.has_edge(actor, movies_list[j]):
                            self.graph.add_edge(actor, movies_list[j], relation="合作出演")

        print(f"知识图谱构建完成:")
        print(f"  节点数: {self.graph.number_of_nodes()}")
        print(f"  边数: {self.graph.number_of_edges()}")

    def query_single_hop(self, entity_name: str) -> List[Tuple[str, str, str]]:
        """
        单跳邻居查询

        Args:
            entity_name: 实体名称

        Returns:
            三元组列表 [(源实体, 关系, 目标实体), ...]
        """
        triples = []

        if entity_name not in self.graph:
            return triples

        # 出边
        for _, target, data in self.graph.out_edges(entity_name, data=True):
            triples.append((entity_name, data.get("relation", ""), target))

        # 入边
        for source, _, data in self.graph.in_edges(entity_name, data=True):
            triples.append((source, data.get("relation", ""), entity_name))

        return triples

    def query_two_hop(self, entity_name: str) -> List[Tuple[str, str, str, str, str]]:
        """
        两跳关系查询

        Args:
            entity_name: 起始实体名称

        Returns:
            五元组列表 [(实体1, 关系1, 中间实体, 关系2, 实体2), ...]
        """
        paths = []

        if entity_name not in self.graph:
            return paths

        # 第一跳
        for _, mid_node, data1 in self.graph.out_edges(entity_name, data=True):
            # 第二跳
            for _, end_node, data2 in self.graph.out_edges(mid_node, data=True):
                if end_node != entity_name:
                    paths.append((
                        entity_name,
                        data1.get("relation", ""),
                        mid_node,
                        data2.get("relation", ""),
                        end_node
                    ))

        return paths

    def query_common_neighbors(self, entity1: str, entity2: str) -> List[str]:
        """
        共同邻居查询

        Args:
            entity1: 实体1
            entity2: 实体2

        Returns:
            共同邻居列表
        """
        if entity1 not in self.graph or entity2 not in self.graph:
            return []

        neighbors1 = set(self.graph.neighbors(entity1))
        neighbors2 = set(self.graph.neighbors(entity2))

        return list(neighbors1 & neighbors2)

    def get_entity_info(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """
        获取实体信息

        Args:
            entity_name: 实体名称

        Returns:
            实体信息字典
        """
        if entity_name not in self.graph:
            return None

        node_data = self.graph.nodes[entity_name]
        return {
            "name": entity_name,
            "type": node_data.get("type", "unknown"),
            **{k: v for k, v in node_data.items() if k != "type"}
        }

    def save(self, path: str = None) -> None:
        """
        保存知识图谱到文件

        Args:
            path: 保存路径
        """
        if path is None:
            from src.utils.common import get_data_dir
            path = get_data_dir() / "kg" / "knowledge_graph.pkl"

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "wb") as f:
            pickle.dump(self.graph, f)
        print(f"知识图谱已保存到: {save_path}")

    def load(self, path: str = None) -> None:
        """
        从文件加载知识图谱

        Args:
            path: 知识图谱文件路径
        """
        if path is None:
            from src.utils.common import get_data_dir
            path = get_data_dir() / "kg" / "knowledge_graph.pkl"

        with open(path, "rb") as f:
            self.graph = pickle.load(f)
        print(f"知识图谱加载完成:")
        print(f"  节点数: {self.graph.number_of_nodes()}")
        print(f"  边数: {self.graph.number_of_edges()}")


def build_knowledge_graph() -> KnowledgeGraph:
    """
    构建知识图谱

    Returns:
        KnowledgeGraph 实例
    """
    kg = KnowledgeGraph()
    kg.build_graph()
    return kg


if __name__ == "__main__":
    print("=" * 50)
    print("NetworkX 知识图谱测试")
    print("=" * 50)

    kg = build_knowledge_graph()

    # 测试单跳查询
    print("\n单跳查询 '阿凡达':")
    triples = kg.query_single_hop("阿凡达")
    for t in triples[:5]:
        print(f"  {t[0]} --[{t[1]}]--> {t[2]}")

    # 测试两跳查询
    print("\n两跳查询 '阿凡达':")
    paths = kg.query_two_hop("阿凡达")
    for p in paths[:3]:
        print(f"  {p[0]} --[{p[1]}]--> {p[2]} --[{p[3]}]--> {p[4]}")

    # 测试共同邻居
    print("\n共同邻居查询 '萨姆·沃辛顿' 和 '西格妮·韦弗':")
    common = kg.query_common_neighbors("萨姆·沃辛顿", "西格妮·韦弗")
    print(f"  共同出演: {common}")
