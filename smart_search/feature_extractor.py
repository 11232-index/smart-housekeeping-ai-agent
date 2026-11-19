"""用 LLM 从目录记录中推断特征标签。

替代写死的 TAG_RULES，避免业务变了标签还停留在代码里。

- 只在建索引时调用，不在用户查询时调用
- 结果按记录内容哈希缓存，内容不变不重复调用
- 失败时返回空特征，不阻塞建库
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from smart_search.config import FEATURE_CACHE_PATH
from smart_search.llm import get_llm

EMPTY_FEATURES: Dict[str, Any] = {
    "tags": [],
    "skills": [],
    "suitable_for": [],
    "summary": "",
}

PROMPT = """你是家政平台的目录标注器。请根据记录信息，推断用户可能用哪些词来找它。

只输出 JSON，不要 Markdown，不要解释：
{{"tags":["..."],"skills":["..."],"suitable_for":["..."],"summary":"一句话概括"}}

规则：
- tags：服务/套餐的特征、适合人群、清洁类型；家政员可留空数组。
- skills：家政员擅长技能；服务/套餐留空数组。
- suitable_for：适合的家庭、人群或场景。
- summary：一句话概括，不超过 40 字。
- 只能基于给定信息推断；信息不足就少写或不写，禁止编造具体服务内容或价格。
- 标签用 2~6 个字的中文短词，最多各 6 个，去重。
- 只输出 JSON。

记录信息：
{record}
"""


def _load_cache() -> Dict[str, Any]:
    path = Path(FEATURE_CACHE_PATH)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(cache: Dict[str, Any]) -> None:
    path = Path(FEATURE_CACHE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _clean_list(value: Any, limit: int = 6) -> List[str]:
    if not isinstance(value, list):
        return []
    result: List[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
        if len(result) >= limit:
            break
    return result


def _parse(raw: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", raw or "", re.S)
    if not match:
        return dict(EMPTY_FEATURES)
    try:
        data = json.loads(match.group(0))
    except Exception:
        return dict(EMPTY_FEATURES)
    return {
        "tags": _clean_list(data.get("tags")),
        "skills": _clean_list(data.get("skills")),
        "suitable_for": _clean_list(data.get("suitable_for")),
        "summary": str(data.get("summary") or "").strip()[:80],
    }


def extract_features(record: Dict[str, Any]) -> Dict[str, Any]:
    """对一条目录记录做特征抽取，带本地缓存。

    输入示例：
        {"kind": "package", "name": "穹妹", "description": "...", "price": 15}

    输出示例：
        {"tags": ["二次元", "主题套餐"],
         "skills": [],
         "suitable_for": ["二次元爱好者"],
         "summary": "春日野穹系列主题套餐"}
    """
    # 把记录序列化成稳定字符串，再算哈希，作为“内容指纹”。
    # 内容不变 → 指纹不变 → 直接复用缓存，不调用 LLM。
    payload_text = json.dumps(record, ensure_ascii=False, sort_keys=True)
    cache_key = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()

    cache = _load_cache()
    if cache_key in cache:
        # 缓存命中：同一个内容之前已经让 LLM 标注过
        return cache[cache_key]

    try:
        # 缓存未命中：调用 LLM，只允许它输出 JSON
        response = get_llm().invoke(PROMPT.format(record=payload_text))
        raw = getattr(response, "content", "") or str(response)
        features = _parse(raw)
    except Exception as exc:
        # LLM 失败不阻塞建库：返回空特征，后续仍可用名称/描述检索
        print(f"⚠️ 特征抽取失败，使用空特征: {exc}")
        features = dict(EMPTY_FEATURES)

    # 写回缓存，下次同样内容就不再花钱调用 LLM
    cache[cache_key] = features
    try:
        _save_cache(cache)
    except Exception as exc:
        print(f"⚠️ 特征缓存写入失败: {exc}")
    return features
