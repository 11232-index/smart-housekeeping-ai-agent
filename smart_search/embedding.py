"""智能搜索专用 embedding 封装（将人类语言转换为向量）。

与客服 knowledge/embeddings.py 使用同一个 SiliconFlow bge-m3，
但代码独立，避免智能搜索影响客服模块。
"""

from functools import lru_cache

from langchain.embeddings import init_embeddings

from smart_search.config import (
    EMBEDDING_MODEL,
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
)


@lru_cache(maxsize=1)
def get_embeddings():
    if not SILICONFLOW_API_KEY:
        raise ValueError("未配置 SILICONFLOW_API_KEY，请在项目根目录 .env 中补充")
    return init_embeddings(
        model=EMBEDDING_MODEL,
        api_key=SILICONFLOW_API_KEY,
        base_url=SILICONFLOW_BASE_URL,
    )
