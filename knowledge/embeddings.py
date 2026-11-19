"""
③ 文档嵌入模型（Embeddings）

职责：
    - 通过 SiliconFlow 的 BAAI/bge-m3 把文本转成 1024 维向量
    - 建库时把知识块批量转成向量
    - 查询时把用户问题转成向量
    - 保证建库与查询使用同一个模型

不负责：
    - 保存向量（vector_store.py）
    - 读写文件（loader.py）
    - 文档切分（splitter.py）

说明：
    本文件是无状态的，不保存任何数据。Chroma 的 add_documents() /
    similarity_search() 会在内部通过 embedding_function 自动调用它。

配置：
    .env 中需要：
        SILICONFLOW_API_KEY=sk-xxx
        SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
    模型名读取 .env 的 SILICONFLOW_MODEL，代码会自动补上 openai: 前缀。
"""

from functools import lru_cache

from langchain.embeddings import init_embeddings

from knowledge.config import (
    EMBEDDING_MODEL,
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
)


@lru_cache(maxsize=1)
def get_embeddings():
    """返回全局唯一的 embedding 客户端。

    Returns:
        Embeddings: LangChain 嵌入模型实例，支持
        embed_documents() 和 embed_query()。

    Raises:
        ValueError: 未配置 SILICONFLOW_API_KEY 时抛出。
    """
    if not SILICONFLOW_API_KEY:
        raise ValueError(
            "未配置 SILICONFLOW_API_KEY，请在项目根目录 .env 中补充"
        )

    return init_embeddings(
        model=EMBEDDING_MODEL,
        api_key=SILICONFLOW_API_KEY,
        base_url=SILICONFLOW_BASE_URL,
    )
