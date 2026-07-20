"""
文档检索 Agent
整合 BM25 检索与向量检索，实现混合召回 + 重排序，返回 Top5 结果
"""

import json
from typing import Dict, List, Any, Optional

from src.utils.common import load_config
from src.utils.chroma_utils import VectorStore
from src.utils.bm25_utils import BM25Retriever


class DocumentRetrieverAgent:
    """文档检索 Agent，整合 BM25 与向量检索，支持重排序"""

    def __init__(
        self,
        vector_store=None,
        bm25_retriever=None,
        use_reranker: bool = True
    ):
        """
        初始化文档检索 Agent

        Args:
            vector_store: 预初始化的 VectorStore 实例，默认自动获取单例
            bm25_retriever: 预初始化的 BM25Retriever 实例，默认自动获取单例
            use_reranker: 是否使用 BGE-Reranker 进行重排序
        """
        from src.utils.chroma_utils import get_vector_store
        from src.utils.bm25_utils import get_bm25_retriever

        self.vector_store = vector_store or get_vector_store()
        self.bm25_retriever = bm25_retriever or get_bm25_retriever()
        self.use_reranker = use_reranker

        # 初始化重排序模型
        self.reranker = None
        if use_reranker:
            self._init_reranker()

    def _init_reranker(self):
        """初始化 BGE-Reranker 模型"""
        try:
            from sentence_transformers import CrossEncoder
            model_name = "BAAI/bge-reranker-v2-m3"
            self.reranker = CrossEncoder(model_name)
            print(f"重排序模型加载成功: {model_name}")
        except Exception as e:
            print(f"重排序模型加载失败: {e}，将使用原始排序")
            self.reranker = None

    def _deduplicate_results(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        对检索结果去重

        Args:
            results: 检索结果列表

        Returns:
            去重后的结果列表
        """
        seen_ids = set()
        unique_results = []

        for result in results:
            doc_id = result.get("id")
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_results.append(result)

        return unique_results

    def _merge_results(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4
    ) -> List[Dict[str, Any]]:
        """
        合并向量检索和 BM25 检索结果

        Args:
            vector_results: 向量检索结果
            bm25_results: BM25 检索结果
            vector_weight: 向量检索权重
            bm25_weight: BM25 检索权重

        Returns:
            合并后的结果列表
        """
        # 为每个结果添加来源标记
        for r in vector_results:
            r["source_type"] = "vector"
        for r in bm25_results:
            r["source_type"] = "bm25"

        # 合并结果
        all_results = vector_results + bm25_results

        # 去重
        unique_results = self._deduplicate_results(all_results)

        # 计算综合分数
        for result in unique_results:
            vector_score = 0.0
            bm25_score = 0.0

            if result["source_type"] == "vector":
                # 向量检索的距离越小越好，转换为相似度
                distance = result.get("distance", 1.0)
                vector_score = 1.0 - distance
            elif result["source_type"] == "bm25":
                # BM25 分数归一化
                bm25_score = result.get("score", 0.0)

            result["combined_score"] = (
                vector_weight * vector_score +
                bm25_weight * bm25_score
            )

        # 按综合分数排序
        unique_results.sort(key=lambda x: x["combined_score"], reverse=True)

        return unique_results

    def _rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        使用 BGE-Reranker 对结果进行重排序

        Args:
            query: 查询文本
            results: 待重排序的结果列表
            top_k: 返回的 top-k 结果数量

        Returns:
            重排序后的结果列表
        """
        if self.reranker is None or len(results) == 0:
            return results[:top_k]

        # 准备重排序输入
        pairs = [(query, r["document"]) for r in results]

        # 计算重排序分数
        scores = self.reranker.predict(pairs)

        # 更新结果分数
        for i, score in enumerate(scores):
            results[i]["rerank_score"] = float(score)

        # 按重排序分数排序
        results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        return results[:top_k]

    def _calculate_confidence(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        计算每个结果的置信度

        Args:
            results: 检索结果列表

        Returns:
            带置信度的结果列表
        """
        if not results:
            return results

        # 获取分数范围
        scores = [r.get("rerank_score", r.get("combined_score", 0)) for r in results]
        max_score = max(scores) if scores else 1.0
        min_score = min(scores) if scores else 0.0
        score_range = max_score - min_score if max_score > min_score else 1.0

        for result in results:
            score = result.get("rerank_score", result.get("combined_score", 0))
            # 归一化到 0.5-1.0 范围
            confidence = 0.5 + 0.5 * (score - min_score) / score_range
            result["confidence"] = round(confidence, 4)

        return results

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        n_vector_results: int = 10,
        n_bm25_results: int = 10
    ) -> Dict[str, Any]:
        """
        执行文档检索

        Args:
            query: 查询文本
            top_k: 返回的 top-k 结果数量
            n_vector_results: 向量检索返回的结果数量
            n_bm25_results: BM25 检索返回的结果数量

        Returns:
            包含检索结果、置信度、溯源信息的字典
        """
        result = {
            "success": False,
            "results": [],
            "source": "doc",
            "query": query,
            "total_candidates": 0,
            "error": None
        }

        try:
            # 1. 向量检索
            vector_results = self.vector_store.query(
                query_text=query,
                n_results=n_vector_results
            )

            # 2. BM25 检索
            bm25_results = []
            if self.bm25_retriever.bm25 is not None:
                bm25_results = self.bm25_retriever.search(
                    query_text=query,
                    n_results=n_bm25_results
                )

            # 3. 合并结果
            merged_results = self._merge_results(
                vector_results,
                bm25_results
            )
            result["total_candidates"] = len(merged_results)

            # 4. 重排序
            reranked_results = self._rerank_results(
                query,
                merged_results,
                top_k=top_k
            )

            # 5. 计算置信度
            final_results = self._calculate_confidence(reranked_results)

            # 6. 格式化输出
            formatted_results = []
            for r in final_results:
                formatted_results.append({
                    "id": r.get("id"),
                    "document": r.get("document"),
                    "metadata": r.get("metadata", {}),
                    "confidence": r.get("confidence", 0.5),
                    "source_type": r.get("source_type", "unknown"),
                    "score": r.get("rerank_score", r.get("combined_score", 0))
                })

            result["success"] = True
            result["results"] = formatted_results

        except Exception as e:
            result["error"] = str(e)

        return result


def create_doc_retriever_agent(
    use_reranker: bool = True,
    vector_store=None,
    bm25_retriever=None
) -> DocumentRetrieverAgent:
    """创建文档检索 Agent 实例

    Args:
        use_reranker: 是否使用重排序
        vector_store: 预初始化的 VectorStore 实例
        bm25_retriever: 预初始化的 BM25Retriever 实例

    Returns:
        DocumentRetrieverAgent 实例
    """
    return DocumentRetrieverAgent(
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
        use_reranker=use_reranker
    )


if __name__ == "__main__":
    print("=" * 60)
    print("文档检索 Agent 测试")
    print("=" * 60)

    agent = create_doc_retriever_agent(use_reranker=False)

    # 测试查询
    test_queries = [
        "阿凡达电影的剧情介绍",
        "科幻电影有哪些",
        "詹姆斯·卡梅隆导演的作品"
    ]

    for query in test_queries:
        print(f"\n查询: {query}")
        print("-" * 40)
        result = agent.retrieve(query, top_k=3)
        print(f"成功: {result['success']}")
        print(f"候选数量: {result['total_candidates']}")
        print(f"返回结果数: {len(result['results'])}")
        for i, r in enumerate(result['results']):
            print(f"\n  结果 {i+1}:")
            print(f"    来源: {r['metadata'].get('source', 'unknown')}")
            print(f"    置信度: {r['confidence']}")
            print(f"    内容预览: {r['document'][:80]}...")
