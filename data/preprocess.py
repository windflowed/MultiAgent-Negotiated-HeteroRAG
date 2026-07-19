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
