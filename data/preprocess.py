"""
文档加载与切分模块
负责批量加载中文 Wikipedia Markdown 文档并进行分块处理
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(doc_dir: str = None) -> List[Document]:
    """
    批量加载指定目录下的 Markdown 文档

    Args:
        doc_dir: 文档目录路径，默认从配置读取

    Returns:
        LangChain Document 对象列表
    """
    if doc_dir is None:
        from src.utils.common import get_data_dir
        doc_dir = get_data_dir() / "documents"

    doc_path = Path(doc_dir)
    if not doc_path.exists():
        raise FileNotFoundError(f"文档目录不存在: {doc_path}")

    documents = []
    md_files = list(doc_path.glob("*.md"))

    for md_file in md_files:
        try:
            content = md_file.read_text(encoding="utf-8")
            if content.strip():
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": md_file.name,
                        "movie_name": md_file.stem
                    }
                )
                documents.append(doc)
        except Exception as e:
            print(f"加载文档失败: {md_file.name}, 错误: {e}")

    print(f"成功加载 {len(documents)} 个文档")
    return documents


def split_documents(
    documents: List[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64
) -> List[Document]:
    """
    将文档列表切分为更小的文本块

    Args:
        documents: LangChain Document 对象列表
        chunk_size: 每个文本块的最大字符数（默认 512）
        chunk_overlap: 相邻文本块的重叠字符数（默认 64）

    Returns:
        切分后的 Document 对象列表
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"切分完成: {len(documents)} 个文档 -> {len(chunks)} 个文本块")
    return chunks


def get_document_chunks(
    doc_dir: str = None,
    chunk_size: int = 512,
    chunk_overlap: int = 64
) -> List[Document]:
    """
    一键加载并切分文档

    Args:
        doc_dir: 文档目录路径
        chunk_size: 文本块大小
        chunk_overlap: 重叠大小

    Returns:
        切分后的 Document 对象列表
    """
    documents = load_documents(doc_dir)
    chunks = split_documents(documents, chunk_size, chunk_overlap)
    return chunks


if __name__ == "__main__":
    print("=" * 50)
    print("文档加载与切分测试")
    print("=" * 50)

    chunks = get_document_chunks()

    if chunks:
        print(f"\n切分结果统计:")
        print(f"  总文本块数: {len(chunks)}")

        # 统计每个来源的文本块数量
        source_counts = {}
        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
        print(f"  涉及文档数: {len(source_counts)}")

        # 显示第一个文本块示例
        print(f"\n示例文本块:")
        print(f"  来源: {chunks[0].metadata['source']}")
        print(f"  电影名: {chunks[0].metadata['movie_name']}")
        print(f"  内容长度: {len(chunks[0].page_content)} 字符")
        print(f"  内容预览: {chunks[0].page_content[:100]}...")


def validate_sql_database() -> dict:
    """
    校验 SQLite 数据库连接

    Returns:
        包含数据库统计信息的字典
    """
    from src.utils.common import SQLiteDatabase

    db = SQLiteDatabase()
    tables = db.get_tables()

    stats = {"tables": len(tables), "movies": 0}

    for table in tables:
        result = db.execute_query(f"SELECT COUNT(*) as count FROM {table}")
        if result:
            count = result[0].get("count", 0)
            if table == "movies":
                stats["movies"] = count

    return stats


def build_all_datastores() -> dict:
    """
    一键构建所有数据存储

    Returns:
        包含构建统计信息的字典
    """
    from src.utils.chroma_utils import build_vector_store
    from src.utils.bm25_utils import BM25Retriever
    from src.utils.kg_utils import build_knowledge_graph

    stats = {}

    # 1. 加载和切分文档
    print("\n[1/3] 加载和切分文档...")
    documents = load_documents()
    chunks = split_documents(documents)
    stats["documents"] = len(documents)
    stats["chunks"] = len(chunks)

    # 2. 构建向量库
    print("\n[2/3] 构建 ChromaDB 向量库...")
    vector_store = build_vector_store(chunks)
    stats["vector_store"] = "已构建"

    # 3. 构建 BM25 索引
    print("\n[3/3] 构建 BM25 关键词索引...")
    bm25_retriever = BM25Retriever()
    bm25_retriever.build_index(chunks)
    stats["bm25"] = "已构建"

    # 4. 构建知识图谱
    print("\n[4/4] 构建 NetworkX 知识图谱...")
    kg = build_knowledge_graph()
    stats["kg_nodes"] = kg.graph.number_of_nodes()
    stats["kg_edges"] = kg.graph.number_of_edges()
    # 保存到 pickle 文件，供运行时加载
    kg.save()
    stats["kg"] = "已构建并已保存"

    return stats


def run_preprocessing():
    """
    主预处理入口函数
    """
    print("=" * 60)
    print("MultiAgent-Negotiated-HeteroRAG 数据预处理")
    print("=" * 60)

    # Step 1: 校验 SQLite 数据库
    print("\n[Step 1] 校验 SQLite 数据库...")
    try:
        sql_stats = validate_sql_database()
        print(f"  [OK] 数据库连接成功")
        print(f"  [OK] 表数量: {sql_stats['tables']}")
        print(f"  [OK] 电影数量: {sql_stats['movies']}")
    except Exception as e:
        print(f"  [FAIL] 数据库校验失败: {e}")
        return

    # Step 2: 构建所有数据存储
    print("\n[Step 2] 构建数据存储...")
    try:
        ds_stats = build_all_datastores()
        print(f"  [OK] 文档数量: {ds_stats['documents']}")
        print(f"  [OK] 文本块数量: {ds_stats['chunks']}")
        print(f"  [OK] 向量库: {ds_stats['vector_store']}")
        print(f"  [OK] BM25索引: {ds_stats['bm25']}")
        print(f"  [OK] 知识图谱节点数: {ds_stats['kg_nodes']}")
        print(f"  [OK] 知识图谱边数: {ds_stats['kg_edges']}")
    except Exception as e:
        print(f"  [FAIL] 数据存储构建失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 完成
    print("\n" + "=" * 60)
    print("预处理完成！")
    print("=" * 60)
    print("\n数据统计:")
    print(f"  电影数量: {sql_stats['movies']}")
    print(f"  文档数量: {ds_stats['documents']}")
    print(f"  文本块数量: {ds_stats['chunks']}")
    print(f"  知识图谱实体数: {ds_stats['kg_nodes']}")


if __name__ == "__main__":
    run_preprocessing()
