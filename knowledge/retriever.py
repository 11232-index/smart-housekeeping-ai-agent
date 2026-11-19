"""
在线检索入口。

流程：
    1. 向量召回：从 Chroma 取 RERANK_CANDIDATES 条候选
    2. 关键词召回：补充包含查询关键词的候选，解决短中文条目漏召回
    3. 重排：调用 SiliconFlow 的 BGE reranker 精排
    4. 关键词加权：rerank 分数 + 关键词重合度，保证“爽约/补偿”这类词排前面
    5. 重排失败时回退到向量结果

职责：
    - 接收用户 query 和 agent_key
    - 调用 vector_store.search() 获取候选
    - 调用 rerank 模型提升短中文条目的命中率
    - 把结果格式化成带分类和出处的资料文本，供 Agent 拼进 Prompt

不负责：
    - 建库、读取原始文件（indexer.py / loader.py）
    - 业务判断（由各 Agent 决定是否调用本模块）
    - 直接操作 Chroma（统一走 vector_store.py）

对外 API：
    search_knowledge(query, agent_key=None, k=TOP_K)
    format_knowledge(pairs)
"""

import json
import re
import urllib.request
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from knowledge.config import (
    KEYWORD_CANDIDATES,
    KEYWORD_WEIGHT,
    RERANK_CANDIDATES,
    RERANK_ENABLED,
    RERANK_MIN_SCORE,
    RERANK_MODEL,
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
    TOP_K,
)
from knowledge.vector_store import get_vector_store, search

# 关键词重合度用，过滤掉过于通用的短词
_STOPWORDS = {
    "怎么", "什么", "多少", "几点", "可以", "需要", "我的", "你们",
    "这个", "那个", "一下", "吗", "呢", "的", "请问", "如何", "为什么",
}


def _ngrams(text: str) -> set:
    """提取 2~4 字的中文/字母数字片段。"""
    cleaned = re.sub(r"[^一-龥A-Za-z0-9]", "", text or "")
    grams = set()
    for size in (2, 3, 4):
        for index in range(len(cleaned) - size + 1):
            gram = cleaned[index:index + size]
            if gram not in _STOPWORDS:
                grams.add(gram)
    return grams


def _keyword_score(query: str, text: str) -> float:
    """查询关键词在文本中的覆盖率，范围 0~1。"""
    query_grams = _ngrams(query)
    if not query_grams:
        return 0.0
    hits = sum(1 for gram in query_grams if gram in (text or ""))
    return hits / len(query_grams)


def _keyword_candidates(
    query: str,
    agent_key: Optional[str],
    limit: int,
) -> List[Tuple[Document, float]]:
    """从 Chroma 全量文档中挑出关键词命中最多的候选。"""
    store = get_vector_store()
    where = {"agent": agent_key} if agent_key else None
    data = store.get(where=where, include=["documents", "metadatas"])
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []

    scored = []
    for text, meta in zip(documents, metadatas):
        score = _keyword_score(query, text)
        if score > 0:
            scored.append((score, Document(page_content=text, metadata=meta or {})))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [(doc, 0.0) for _, doc in scored[:limit]]


def _rerank(
    query: str,
    pairs: List[Tuple[Document, float]],
    top_n: int,
) -> List[Tuple[Document, float]]:
    """调用 SiliconFlow rerank 接口对候选知识块重新排序。"""
    if not pairs:
        return []

    payload = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": [doc.page_content for doc, _ in pairs],
        "top_n": top_n,
        "return_documents": False,
    }
    request = urllib.request.Request(
        f"{SILICONFLOW_BASE_URL.rstrip('/')}/rerank",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.load(response)

    ranked: List[Tuple[Document, float]] = []
    for item in data.get("results", []):
        index = item.get("index")
        score = float(item.get("relevance_score", 0.0))
        if not isinstance(index, int) or not (0 <= index < len(pairs)):
            continue
        if score < RERANK_MIN_SCORE:
            continue
        ranked.append((pairs[index][0], score))

    ranked.sort(key=lambda pair: pair[1], reverse=True)
    return ranked[:top_n]


def search_knowledge(
    query: str,
    agent_key: Optional[str] = None,
    k: int = TOP_K,
) -> List[Tuple[Document, float]]:
    """检索知识库：向量召回 + 关键词召回 + rerank 精排。"""
    # 1. 向量召回
    candidates = search(
        query,
        agent_key=agent_key,
        k=RERANK_CANDIDATES,
        min_score=0.0,
    )

    if RERANK_ENABLED and SILICONFLOW_API_KEY:
        # 2. 关键词召回，补齐向量漏掉的条目
        keyword_candidates = _keyword_candidates(query, agent_key, KEYWORD_CANDIDATES)
        merged = {}
        for doc, score in candidates + keyword_candidates:
            key = doc.metadata.get("chunk_id") or doc.page_content
            merged.setdefault(key, (doc, score))
        candidates = list(merged.values())

        # 3. rerank
        try:
            reranked = _rerank(query, candidates, max(k * 4, 12))
            if not reranked:
                return []

            # 4. rerank 分数 + 关键词覆盖度加权
            boosted = [
                (doc, score + KEYWORD_WEIGHT * _keyword_score(query, doc.page_content))
                for doc, score in reranked
            ]
            boosted.sort(key=lambda pair: pair[1], reverse=True)
            return boosted[:k]
        except Exception as e:
            print(f"⚠️ 重排失败，回退向量结果: {e}")

    return candidates[:k]


def format_knowledge(pairs: List[Tuple[Document, float]]) -> str:
    """把检索结果格式化成 Prompt 可用的文本。"""
    if not pairs:
        return ""

    blocks = []
    for index, (doc, _score) in enumerate(pairs, 1):
        meta = doc.metadata
        category = meta.get("category", "")
        source = meta.get("source", "")
        blocks.append(
            f"[资料{index}｜{category}｜{source}]\n{doc.page_content}"
        )

    return "\n\n".join(blocks)
