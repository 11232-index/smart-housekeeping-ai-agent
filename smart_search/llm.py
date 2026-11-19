"""智能搜索专用 LLM 客户端。

与客服的 LLM 初始化代码解耦，只复用 .env 中的 OPENAI_* 配置。
"""

from functools import lru_cache

from langchain_openai import ChatOpenAI

from smart_search.config import (
    LLM_TIMEOUT,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
)


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    if not OPENAI_API_KEY:
        raise ValueError("未配置 OPENAI_API_KEY，请在项目根目录 .env 中补充")
    return ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=0,
        timeout=LLM_TIMEOUT,
        max_retries=2,
    )
