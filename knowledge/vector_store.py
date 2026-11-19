"""
④ 向量存储（Vector Store）

职责：
    - 创建/连接 Chroma 持久化集合（data/chroma）
    - 写入知识块（向量由 embeddings.py 自动生成）
    - 按 source 删除旧块，支持增量更新
    - 相似度检索，支持 agent 元数据过滤和分数阈值

不负责：
    - 读取原始文件（loader.py）
    - 文档切分（splitter.py）
    - 选择/初始化 embedding 模型（embeddings.py）
    - 把结果拼成 Prompt 文本（retriever.py）

说明：
    本文件是知识库中唯一有状态的一层，数据、索引和磁盘目录都由它管理。
"""

from functools import lru_cache
from typing import List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document

from knowledge.config import (
    CHROMA_DIR,
    COLLECTION,
    MIN_SCORE,
    TOP_K,
)
from knowledge.embeddings import get_embeddings


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """返回 Chroma 向量库单例（持久化到 data/chroma）。

    Returns:
        Chroma: 绑定 bge-m3 embedding_function 的向量库实例。
    """
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
        # 文本向量用余弦相似度更合适
        collection_metadata={"hnsw:space": "cosine"},
    )


def add_documents(chunks: List[Document]) -> None:
    """把知识块写入向量库。

    Args:
        chunks: splitter 输出的知识块，metadata 中必须包含 chunk_id。
    """
    if not chunks:
        print("⚠️ 没有可写入的知识块")
        return

    store = get_vector_store()
    ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    store.add_documents(chunks, ids=ids)
    print(f"📥 已写入 {len(chunks)} 个知识块")


def delete_by_source(source: str) -> None:
    """按来源文件删除旧知识块（增量更新用）。

    Args:
        source: 相对路径，如 service/日常保洁.md。
    """
    if not source:
        return
    get_vector_store().delete(where={"source": source})


def reset_collection() -> None:
    """清空整个集合（全量重建用）。"""
    get_vector_store().reset_collection()
    print("🧹 向量集合已清空")


def search(
    query: str,
    agent_key: Optional[str] = None,
    k: int = TOP_K,
    min_score: float = MIN_SCORE,
) -> List[Tuple[Document, float]]:
    """相似度检索。

    Args:
        query: 用户查询。
        agent_key: 非空时只在该业务域内检索（metadata filter）；
                   None 表示跨域检索全部知识。
        k: 返回条数。
        min_score: 最低相关度分数，低于该值的结果丢弃。

    Returns:
        List[Tuple[Document, float]]: (知识块, 分数) 列表。
    """
    where = {"agent": agent_key} if agent_key else None

    pairs = get_vector_store().similarity_search_with_relevance_scores(
        query,
        k=k,
        filter=where,
    )
    return [(doc, score) for doc, score in pairs if score >= min_score]
