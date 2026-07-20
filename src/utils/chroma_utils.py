"""
ChromaDB 向量库工具模块
负责文档向量化、入库与相似度检索
"""

from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from src.utils.common import load_config, get_chroma_db_dir


class VectorStore:
    """ChromaDB 向量库封装类"""

    def __init__(self, collection_name: str = "documents"):
        """
        初始化向量库

        Args:
            collection_name: 集合名称
        """
        config = load_config()
        self.db_path = Path(config["CHROMA_DB_DIR"])
        self.db_path.mkdir(parents=True, exist_ok=True)

        # 使用 BGE-small-zh-v1.5 中文优化嵌入模型（512维向量）
        # - 选型理由：专为中文语义检索优化，CPU 即可运行，适合本地部署
        # - 接入方式：通过 ChromaDB 内置的 SentenceTransformerEmbeddingFunction 封装，
        #   无需手动加载 sentence-transformers 模型，ChromaDB 自动管理嵌入生成
        # - 距离度量：cosine 余弦距离（hnsw:space = cosine）
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-small-zh-v1.5",
            device="cpu"
        )

        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = self.client.get_or_create_collection(
            collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ) -> int:
        """
        批量添加文档到向量库

        Args:
            documents: 文档内容列表
            metadatas: 元数据列表
            ids: 文档 ID 列表

        Returns:
            添加的文档数量
        """
        batch_size = 100
        total_added = 0

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_metas = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]

            self.collection.add(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids
            )
            total_added += len(batch_docs)

        return total_added

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        相似度查询

        Args:
            query_text: 查询文本
            n_results: 返回结果数量
            where: 过滤条件

        Returns:
            查询结果列表
        """
        query_params = {
            "query_texts": [query_text],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"]
        }
        if where:
            query_params["where"] = where

        results = self.collection.query(**query_params)

        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })

        return formatted_results

    def get_count(self) -> int:
        """
        获取向量库中的文档数量

        Returns:
            文档数量
        """
        return self.collection.count()

    def delete_all(self):
        """清空向量库"""
        all_ids = self.collection.get()["ids"]
        if all_ids:
            self.collection.delete(ids=all_ids)


def build_vector_store(chunks) -> VectorStore:
    """
    从文档块构建向量库

    Args:
        chunks: LangChain Document 对象列表

    Returns:
        VectorStore 实例
    """
    store = VectorStore()

    documents = []
    metadatas = []
    ids = []

    for i, chunk in enumerate(chunks):
        documents.append(chunk.page_content)
        metadatas.append({
            "source": chunk.metadata.get("source", "unknown"),
            "movie_name": chunk.metadata.get("movie_name", "unknown"),
            "chunk_idx": i
        })
        ids.append(f"doc_{i}")

    count = store.add_documents(documents, metadatas, ids)
    print(f"向量库构建完成，共添加 {count} 个文档块")
    return store


def search_similar(
    query_text: str,
    n_results: int = 5
) -> List[Dict[str, Any]]:
    """
    搜索相似文档

    Args:
        query_text: 查询文本
        n_results: 返回结果数量

    Returns:
        查询结果列表
    """
    store = VectorStore()
    return store.query(query_text, n_results)


if __name__ == "__main__":
    print("=" * 50)
    print("ChromaDB 向量库测试")
    print("=" * 50)

    from data.preprocess import get_document_chunks

    chunks = get_document_chunks()
    if chunks:
        store = build_vector_store(chunks)
        print(f"\n向量库文档数量: {store.get_count()}")

        results = search_similar("阿凡达电影", n_results=3)
        print(f"\n查询 '阿凡达电影' 的结果:")
        for r in results:
            print(f"  来源: {r['metadata']['source']}")
            print(f"  距离: {r['distance']:.4f}")
            print(f"  内容: {r['document'][:100]}...")
            print()


# 模块级单例：向量库
_vector_store_singleton = None


def get_vector_store(collection_name: str = "documents") -> VectorStore:
    """
    获取向量库单例，延迟加载

    Args:
        collection_name: 集合名称

    Returns:
        VectorStore 单例实例
    """
    global _vector_store_singleton
    if _vector_store_singleton is None:
        _vector_store_singleton = VectorStore(collection_name)
    return _vector_store_singleton
