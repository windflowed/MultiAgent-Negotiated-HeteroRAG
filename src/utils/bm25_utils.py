"""
BM25 关键词检索模块
基于 rank-bm25 实现中文文档的关键词检索
"""

import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

import jieba
from rank_bm25 import BM25Okapi


def tokenize_chinese(text: str) -> List[str]:
    """
    中文分词函数

    Args:
        text: 中文文本

    Returns:
        分词后的词列表
    """
    tokens = jieba.lcut(text)
    tokens = [t.strip() for t in tokens if t.strip()]
    return tokens


class BM25Retriever:
    """BM25 关键词检索器"""

    def __init__(self):
        """初始化检索器"""
        self.documents = []
        self.metadatas = []
        self.ids = []
        self.bm25 = None
        self.tokenized_corpus = None

    def build_index(self, chunks) -> None:
        """
        从文档块构建 BM25 索引

        Args:
            chunks: LangChain Document 对象列表
        """
        self.documents = []
        self.metadatas = []
        self.ids = []

        for i, chunk in enumerate(chunks):
            self.documents.append(chunk.page_content)
            self.metadatas.append({
                "source": chunk.metadata.get("source", "unknown"),
                "movie_name": chunk.metadata.get("movie_name", "unknown"),
                "chunk_idx": i
            })
            self.ids.append(f"doc_{i}")

        self.tokenized_corpus = [tokenize_chinese(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"BM25 索引构建完成，共 {len(self.documents)} 个文档块")

    def search(
        self,
        query_text: str,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        关键词检索

        Args:
            query_text: 查询文本
            n_results: 返回结果数量

        Returns:
            检索结果列表，格式与向量检索对齐
        """
        if self.bm25 is None:
            raise RuntimeError("请先调用 build_index() 构建索引")

        tokenized_query = tokenize_chinese(query_text)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = scores.argsort()[::-1][:n_results]

        results = []
        for idx in top_indices:
            results.append({
                "id": self.ids[idx],
                "document": self.documents[idx],
                "metadata": self.metadatas[idx],
                "score": float(scores[idx])
            })

        return results

    def save(self, path: str = None) -> None:
        """
        保存 BM25 索引到文件

        Args:
            path: 保存路径
        """
        if path is None:
            from src.utils.common import get_data_dir
            path = get_data_dir() / "bm25_index.pkl"

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "metadatas": self.metadatas,
                "ids": self.ids,
                "tokenized_corpus": self.tokenized_corpus
            }, f)
        print(f"BM25 索引已保存到: {save_path}")

    def load(self, path: str = None) -> None:
        """
        从文件加载 BM25 索引

        Args:
            path: 索引文件路径
        """
        if path is None:
            from src.utils.common import get_data_dir
            path = get_data_dir() / "bm25_index.pkl"

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.documents = data["documents"]
        self.metadatas = data["metadatas"]
        self.ids = data["ids"]
        self.tokenized_corpus = data["tokenized_corpus"]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"BM25 索引加载完成，共 {len(self.documents)} 个文档块")


def build_bm25_index(chunks) -> BM25Retriever:
    """
    从文档块构建 BM25 索引

    Args:
        chunks: LangChain Document 对象列表

    Returns:
        BM25Retriever 实例
    """
    retriever = BM25Retriever()
    retriever.build_index(chunks)
    return retriever


def search_bm25(
    query_text: str,
    n_results: int = 5,
    index_path: str = None
) -> List[Dict[str, Any]]:
    """
    BM25 关键词检索

    Args:
        query_text: 查询文本
        n_results: 返回结果数量
        index_path: 索引文件路径

    Returns:
        检索结果列表
    """
    retriever = BM25Retriever()
    retriever.load(index_path)
    return retriever.search(query_text, n_results)


if __name__ == "__main__":
    print("=" * 50)
    print("BM25 关键词检索测试")
    print("=" * 50)

    from data.preprocess import get_document_chunks

    chunks = get_document_chunks()
    if chunks:
        retriever = build_bm25_index(chunks)

        results = retriever.search("阿凡达电影", n_results=3)
        print(f"\n查询 '阿凡达电影' 的结果:")
        for r in results:
            print(f"  来源: {r['metadata']['source']}")
            print(f"  分数: {r['score']:.4f}")
            print(f"  内容: {r['document'][:100]}...")
            print()


# 模块级单例：BM25 检索器
_bm25_retriever_singleton = None


def get_bm25_retriever() -> BM25Retriever:
    """
    获取 BM25 检索器单例，延迟加载

    Returns:
        BM25Retriever 单例实例
    """
    global _bm25_retriever_singleton
    if _bm25_retriever_singleton is None:
        _bm25_retriever_singleton = BM25Retriever()
        _bm25_retriever_singleton.load()
    return _bm25_retriever_singleton


# 模块级单例：BM25 检索器
_bm25_retriever_singleton = None


def get_bm25_retriever() -> BM25Retriever:
    """
    获取 BM25 检索器单例，延迟加载

    Returns:
        BM25Retriever 单例实例
    """
    global _bm25_retriever_singleton
    if _bm25_retriever_singleton is None:
        _bm25_retriever_singleton = BM25Retriever()
        _bm25_retriever_singleton.load()
    return _bm25_retriever_singleton
