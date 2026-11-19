"""智能搜索工具（独立于智能客服的 TOOLS_BY_AGENT）。

每个工具的链路：
    精确名匹配 → 向量召回 → rerank → MySQL 回表 → 过滤 → JSON

工具只返回真实业务库数据；Qdrant 只负责召回。
"""

import datetime as _dt
import json
import re
from typing import Any, Dict, List

from langchain_core.tools import tool
from qdrant_client import models

from smart_search.config import (
    HOUSEKEEPERS_COLLECTION,
    RECALL_LIMIT,
    SERVICES_COLLECTION,
)
from smart_search.embedding import get_embeddings
from smart_search.qdrant_store import get_client
from smart_search.rerank import rerank
from smart_search.repository import (
    fetch_categories,
    fetch_housekeepers_by_ids,
    fetch_housekeepers_exact,
    fetch_services_by_category,
    fetch_services_by_ids,
    fetch_services_exact,
    fetch_slots,
    list_all_services,
)


def _ok(data: List[Dict[str, Any]]) -> str:
    return json.dumps({"ok": True, "data": data}, ensure_ascii=False)


def _err(message: str) -> str:
    return json.dumps({"ok": False, "error": message}, ensure_ascii=False)


def _service_text(payload: Dict[str, Any]) -> str:
    return " ".join([
        payload.get("name", ""),
        payload.get("category", ""),
        " ".join(payload.get("tags") or []),
        payload.get("summary", ""),
    ])


def _housekeeper_text(payload: Dict[str, Any]) -> str:
    return " ".join([
        payload.get("name", ""),
        " ".join(payload.get("skills") or []),
        " ".join(payload.get("tags") or []),
        payload.get("summary", ""),
    ])


_LIST_ALL_TRIGGERS = ("所有", "全部", "列出", "有哪些", "有什么", "都有哪些", "看看")
_LIST_ALL_FILLERS = (
    "帮我", "帮忙", "请", "请问", "麻烦", "给我", "我想", "我要", "想要",
    "能不能", "能否", "可不可以", "可以", "想", "要",
    "输出", "列出", "所有", "全部", "的", "一下", "看看",
    "相关", "服务", "套餐", "项目", "有哪些", "有什么", "都有哪些", "都", "是",
    "和", "与", "及",
)


def _list_all_kind(query: str) -> str:
    """判断是否在问“列出全部服务/套餐”。

    返回 "service" / "package" / "both"；不是 list-all 则返回空字符串。
    去掉通用词后如果还剩特征描述（如“适合养宠物”），就不算 list-all。
    """
    text = (query or "").strip()
    if not any(word in text for word in _LIST_ALL_TRIGGERS):
        return ""
    if not any(word in text for word in ("服务", "套餐", "项目")):
        return ""

    residue = text
    for word in _LIST_ALL_FILLERS:
        residue = residue.replace(word, "")
    residue = re.sub(r"[\s,，。！？?、：:；;\"'“”‘’]", "", residue)
    if len(residue) > 1:
        return ""

    wants_service = ("服务" in text) or ("项目" in text)
    wants_package = "套餐" in text
    if wants_service and wants_package:
        return "both"
    if wants_package:
        return "package"
    return "service"


def _detect_category(query: str) -> str:
    """从问题里识别库内分类名，例如“春日野穹系列”“韩湘子系列”。

    只有问题明显在问“分类/系列/有哪些/全部”时才启用，
    避免把“韩湘子服务”这类具体服务名误当成分类浏览。
    """
    text = (query or "").strip()
    if not text:
        return ""
    if not any(word in text for word in ("分类", "类别", "系列", "哪些", "全部", "所有")):
        return ""
    try:
        categories = fetch_categories()
    except Exception:
        return ""
    # 长分类名优先，避免“春日野穹系列”被更短的词抢先
    for name in sorted(categories, key=len, reverse=True):
        base = name[:-2] if name.endswith("系列") else name
        if name in text or (len(base) >= 2 and base in text):
            return name
    return ""


def _service_result(row: Dict[str, Any], reason: str, score: float = 0.0) -> Dict[str, Any]:
    return {
        "type": row["kind"],
        "kind": row["kind"],
        "key": f'{row["kind"]}:{row["id"]}',
        "id": row["id"],
        "name": row["name"],
        "price": row["price"],
        "image": row.get("image", ""),
        "description": row.get("description", ""),
        "category": row.get("category", ""),
        "requirements": row.get("requirements", []),
        "available_slots": [],
        "reason": reason,
        "score": round(float(score or 0), 4),
    }


def _housekeeper_result(
    row: Dict[str, Any],
    slots: List[str],
    reason: str,
    score: float = 0.0,
) -> Dict[str, Any]:
    return {
        "type": "housekeeper",
        "kind": "housekeeper",
        "key": f'housekeeper:{row["id"]}',
        "id": row["id"],
        "name": row["name"],
        "price": row["price"],
        "rating": row["rating"],
        "rank": row["rank"],
        "rank_text": {1: "金牌", 2: "银牌", 3: "铜牌"}.get(row["rank"], "普通"),
        "picture": row.get("picture", ""),
        "carousel": row.get("carousel", ""),
        "education": row.get("education", ""),
        "position": row.get("position", ""),
        "past_experience": row.get("past_experience", ""),
        "work_status": row.get("work_status", ""),
        "health_certificate": row.get("health_certificate", ""),
        "review": row.get("review", ""),
        "available_slots": slots,
        "available": bool(slots),
        "reason": reason,
        "score": round(float(score or 0), 4),
    }


@tool
def search_services(
    query: str,
    max_price: float = 0.0,
    category: str = "",
    list_all: str = "",
) -> str:
    """搜索家政服务或服务套餐，也支持按分类列出全部服务。

    优先按名称精确匹配，未命中时进行语义搜索；结果会回 MySQL 校验
    最新名称、价格和上架状态。如果问题里出现库内分类名，或显式传入
    category，则直接列出该分类下的全部上架服务/套餐。

    Args:
        query: 用户需求或服务名称，例如“适合养宠物的保洁”“穹妹”。
        max_price: 最高价格，0 表示不限。
        category: 分类名，例如“春日野穹系列”；留空时自动从 query 识别。
        list_all: service / package / both，表示列出全部；none 或留空表示不启用。
            通常由 Agent 的 LLM 判断后传入，规则识别只作兜底。

    Returns:
        JSON 字符串：{"ok": true, "data": [...]} 或 {"ok": false, "error": "..."}。
    """
    try:
        query = (query or "").strip()
        if not query:
            return _err("请提供要搜索的服务或需求")

        # 1. 分类浏览优先：用户明确点名分类时，列出该分类下全部服务/套餐
        category_name = (category or "").strip() or _detect_category(query)
        if category_name:
            rows = fetch_services_by_category(category_name)
            data = [
                _service_result(row, f"分类：{category_name}")
                for row in rows
                if not max_price or row["price"] <= float(max_price)
            ]
            if data:
                return _ok(data)
            return _err(f"分类“{category_name}”下暂无上架服务或套餐")

        # 2. 明确“列出全部服务/套餐”：直接查库，不走向量检索。
        # list_all 由 Agent 的 LLM 判断；规则识别只作兜底。
        list_kind = (list_all or "").strip().lower()
        if list_kind not in ("service", "package", "both"):
            list_kind = _list_all_kind(query)
        if list_kind:
            rows = list_all_services(list_kind)
            label = {
                "service": "全部服务",
                "package": "全部套餐",
            }.get(list_kind, "全部服务与套餐")
            data = [
                _service_result(row, label)
                for row in rows
                if not max_price or row["price"] <= float(max_price)
            ]
            if data:
                return _ok(data)
            return _err("当前没有上架的服务或套餐")

        # 3. 精确名优先：库内真实名称不交给向量猜
        exact_rows = fetch_services_exact(query)
        if exact_rows:
            data = [
                _service_result(row, "名称精确匹配")
                for row in exact_rows
                if not max_price or row["price"] <= float(max_price)
            ]
            if data:
                return _ok(data)

        # 4. 向量召回
        must = [models.FieldCondition(
            key="status", match=models.MatchValue(value=1)
        )]
        if max_price and float(max_price) > 0:
            must.append(models.FieldCondition(
                key="price", range=models.Range(lte=float(max_price))
            ))
        points = get_client().query_points(
            collection_name=SERVICES_COLLECTION,
            query=get_embeddings().embed_query(query),
            query_filter=models.Filter(must=must),
            limit=RECALL_LIMIT,
        ).points

        candidates = [
            {
                "kind": p.payload.get("kind", ""),
                "id": int(p.payload.get("record_id") or 0),
                "score": float(p.score or 0),
                "text": _service_text(p.payload),
            }
            for p in points
            if p.payload.get("record_id")
        ]

        # 5. rerank 精排
        candidates = rerank(query, candidates)

        # 6. 回表校验最新数据
        ids_by_kind: Dict[str, List[int]] = {}
        for item in candidates:
            ids_by_kind.setdefault(item["kind"], []).append(item["id"])
        rows = fetch_services_by_ids(ids_by_kind)

        data: List[Dict[str, Any]] = []
        seen = set()
        for item in candidates:
            key = f"{item['kind']}:{item['id']}"
            row = rows.get(key)
            if not row or row["status"] != 1:
                continue
            if max_price and float(max_price) > 0 and row["price"] > float(max_price):
                continue
            if key in seen:
                continue
            seen.add(key)
            data.append(_service_result(row, "语义匹配", item.get("score", 0)))

        if not data:
            return _err("没有找到匹配的服务或套餐")
        return _ok(data)
    except Exception as exc:
        return _err(f"搜索服务失败: {exc}")


@tool
def search_housekeepers(
    query: str,
    min_rating: float = 0.0,
    max_price: float = 0.0,
    date: str = "",
) -> str:
    """搜索家政员并返回可约时段。

    优先按姓名精确匹配，未命中时进行语义搜索；档期默认只返回
    明天起 3 个自然日内的可约时段。

    Args:
        query: 用户需求或家政员姓名，例如“会做饭的阿姨”“吕洞宾”。
        min_rating: 最低评分，0 表示不限。
        max_price: 最高单价，0 表示不限。
        date: 指定日期 YYYY-MM-DD，留空表示明天起 3 天。

    Returns:
        JSON 字符串：{"ok": true, "data": [...]} 或 {"ok": false, "error": "..."}。
    """
    try:
        query = (query or "").strip()
        if not query:
            return _err("请提供要找的家政员需求")

        date_from = None
        date_to = None
        if date:
            try:
                date_from = _dt.date.fromisoformat(date)
                date_to = date_from + _dt.timedelta(days=1)
            except ValueError:
                return _err("日期格式应为 YYYY-MM-DD")

        # 1. 精确名优先
        exact_rows = fetch_housekeepers_exact(query)
        if exact_rows:
            slots_map = fetch_slots([r["id"] for r in exact_rows], date_from, date_to)
            data = [
                _housekeeper_result(r, slots_map.get(r["id"], []), "姓名精确匹配")
                for r in exact_rows
                if (not min_rating or r["rating"] >= float(min_rating))
                and (not max_price or r["price"] <= float(max_price))
            ]
            if data:
                return _ok(data)

        # 2. 向量召回
        must = [models.FieldCondition(
            key="status", match=models.MatchValue(value=1)
        )]
        if min_rating and float(min_rating) > 0:
            must.append(models.FieldCondition(
                key="rating", range=models.Range(gte=float(min_rating))
            ))
        if max_price and float(max_price) > 0:
            must.append(models.FieldCondition(
                key="price", range=models.Range(lte=float(max_price))
            ))
        points = get_client().query_points(
            collection_name=HOUSEKEEPERS_COLLECTION,
            query=get_embeddings().embed_query(query),
            query_filter=models.Filter(must=must),
            limit=RECALL_LIMIT,
        ).points

        candidates = [
            {
                "id": int(p.payload.get("record_id") or 0),
                "score": float(p.score or 0),
                "text": _housekeeper_text(p.payload),
            }
            for p in points
            if p.payload.get("record_id")
        ]

        # 3. rerank 精排
        candidates = rerank(query, candidates)

        # 4. 回表校验
        rows = fetch_housekeepers_by_ids([c["id"] for c in candidates])
        slots_map = fetch_slots(list(rows.keys()), date_from, date_to)

        data: List[Dict[str, Any]] = []
        seen = set()
        for item in candidates:
            row = rows.get(item["id"])
            if not row or row["status"] != 1:
                continue
            if min_rating and float(min_rating) > 0 and row["rating"] < float(min_rating):
                continue
            if max_price and float(max_price) > 0 and row["price"] > float(max_price):
                continue
            if item["id"] in seen:
                continue
            seen.add(item["id"])
            data.append(_housekeeper_result(
                row,
                slots_map.get(item["id"], []),
                "语义匹配",
                item.get("score", 0),
            ))

        if not data:
            return _err("没有找到匹配的家政员")
        return _ok(data)
    except Exception as exc:
        return _err(f"搜索家政员失败: {exc}")


# 只给智能搜索 Agent 使用，不加入客服的 TOOLS_BY_AGENT
SMART_SEARCH_TOOLS = [search_services, search_housekeepers]
