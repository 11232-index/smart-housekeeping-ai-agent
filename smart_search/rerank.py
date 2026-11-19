"""SiliconFlow bge-reranker 精排。

Qdrant 向量召回只负责“找得全”，本模块负责“排得准”。
接口失败时自动退回向量分排序，不阻塞搜索。
"""

import json
import urllib.request
from typing import Any, Dict, List

from smart_search.config import (
    RERANK_MIN_SCORE,
    RERANK_MODEL,
    RERANK_TOP_N,
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
)


def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_n: int = RERANK_TOP_N,
) -> List[Dict[str, Any]]:
    """对候选重新打分。

    candidates 每项至少包含：
        text  —— 用于 rerank 的文本
        score —— Qdrant 向量分（接口失败时作为兜底排序）
    """
    if not candidates:
        return []
    if not SILICONFLOW_API_KEY:
        return sorted(candidates, key=lambda c: c.get("score", 0), reverse=True)[:top_n]

    payload = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": [c.get("text", "") for c in candidates],
        "top_n": top_n,
        "return_documents": False,
    }

    try:
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

        ranked: List[Dict[str, Any]] = []
        for item in data.get("results", []):
            index = item.get("index")
            score = float(item.get("relevance_score", 0.0))
            if not isinstance(index, int) or not (0 <= index < len(candidates)):
                continue
            if score < RERANK_MIN_SCORE:
                continue
            candidate = dict(candidates[index])
            candidate["rerank_score"] = score
            ranked.append(candidate)

        ranked.sort(key=lambda c: c.get("rerank_score", 0), reverse=True)
        # 接口成功但没有候选达到阈值：视为“没有足够匹配”，
        # 不要退回向量分，否则会把不相关结果当推荐返回。
        return ranked[:top_n]
    except Exception as exc:
        print(f"⚠️ rerank 失败，退回向量分排序: {exc}")
        return sorted(candidates, key=lambda c: c.get("score", 0), reverse=True)[:top_n]
