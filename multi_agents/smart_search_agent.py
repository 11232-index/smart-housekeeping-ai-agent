"""智能搜索 Agent（独立于智能客服的 6+1 个专家）。

职责：
- 理解用户自然语言需求
- 调用智能搜索工具
- 只基于真实工具结果生成回答

不负责：
- 订单、支付、投诉、售后
- 客服会话记忆、规则 RAG、客服路由
"""

import json
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from smart_search.llm import get_llm
from smart_search.repository import fetch_housekeepers_exact, fetch_services_exact
from tools.smart_search_tools import (
    _list_all_kind,
    search_housekeepers,
    search_services,
)

_JSON_RE = re.compile(r"\{.*\}", re.S)

# 问题里出现这些特征词时，即使 LLM 判了 list_all，也要退回语义搜索，
# 避免把“适合养宠物的所有服务”误变成“列出全部服务”。
_FEATURE_WORDS = (
    "适合", "需要", "不要", "超过", "低于", "以内", "以上",
    "预算", "价格", "评分", "宠物", "做饭", "老人", "月嫂",
    "育儿", "保洁", "清洗", "维修", "深度", "除螨", "陪护",
)

_UNDERSTAND_PROMPT = """你是智能家政搜索的需求解析器。请阅读用户问题，输出严格 JSON：

{{"target":"service","max_price":0,"min_rating":0,"date":"","category":"","list_all":"none","keywords":""}}

字段说明：
- target：service（找服务/套餐）、housekeeper（找家政员）、both（两者都找）
- max_price：最高价格，0 表示未提及
- min_rating：最低评分，0 表示未提及
- date：指定日期 YYYY-MM-DD，未提及留空
- category：用户点名的分类/系列，例如“春日野穹系列”；未提及留空
- list_all：service / package / both / none。用户想“列出全部、所有、完整清单、罗列一下、都有啥”时填对应类型；带具体特征、预算、分类条件时填 none
- keywords：用于搜索的核心词，保留服务特征或姓名

规则：
- 用户说“阿姨/家政员/保姆/月嫂/育儿嫂/陪护” → housekeeper
- 用户说“服务/套餐/保洁/清洗/维修” → service
- 同时提到两类 → both
- 用户问“某分类/某系列下有哪些服务” → target=service，category 填该分类名
- keywords 去掉“输出/列出/所有/全部/帮我/看看/一下/相关”等虚词，只保留核心名词
- list_all 判断优先于 keywords：明确要列出全部时，list_all 填对应类型，keywords 保留原问题即可
- category 有值时 list_all 必须填 none，表示“列出该分类”，不是“列出全部”
- 只输出 JSON，不要 Markdown，不要解释。

用户问题：{query}
"""

_ANSWER_PROMPT = """你是智能家政搜索顾问。请只根据下面的【搜索结果】回答用户。

只输出 JSON，不要 Markdown，不要解释：
{{"answer":"给用户的自然语言回答","selected_ids":["package:42"]}}

规则：
- answer 只能使用【搜索结果】里的名称、价格、描述、评分、技能、可约时段。
- 禁止补充搜索结果里没有的服务内容、价格、技能或承诺。
- 如果结果为空，answer 直接说“没有找到匹配的服务/家政员”，selected_ids 返回 []。
- answer 简洁，先给结论，再列匹配项和理由。
- selected_ids 只能从【搜索结果】的 key 字段里选，不能编造。
- answer 里提到几项，selected_ids 就选几项；完全不匹配的不要选。
- 没有匹配项时 selected_ids 返回空数组。

【搜索结果】
{results}

用户问题：{query}
"""


def _parse_json(raw: str) -> Dict[str, Any]:
    match = _JSON_RE.search(raw or "")
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


def _fallback_target(query: str) -> str:
    housekeeper_words = ("阿姨", "家政员", "保姆", "月嫂", "育儿嫂", "陪护", "钟点工")
    service_words = ("服务", "套餐", "保洁", "清洗", "维修", "疏通", "除螨")
    has_hk = any(w in query for w in housekeeper_words)
    has_service = any(w in query for w in service_words)
    if has_hk and has_service:
        return "both"
    if has_hk:
        return "housekeeper"
    return "service"


class SmartSearchAgent:
    """智能搜索 Agent：需求理解 → 工具 → 真实结果回答。"""

    def __init__(self, llm=None):
        self.llm = llm or get_llm()

    def _understand(self, query: str, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        filters = filters or {}
        decision: Dict[str, Any] = {
            "target": _fallback_target(query),
            "max_price": float(filters.get("max_price") or 0),
            "min_rating": float(filters.get("min_rating") or 0),
            "date": str(filters.get("date") or ""),
            "category": str(filters.get("category") or ""),
            "list_all": "",
            "keywords": query,
        }
        try:
            response = self.llm.invoke([
                SystemMessage(content=_UNDERSTAND_PROMPT.format(query=query)),
                HumanMessage(content=query),
            ])
            data = _parse_json(getattr(response, "content", "") or str(response))
            target = str(data.get("target") or "").strip().lower()
            if target in ("service", "housekeeper", "both"):
                decision["target"] = target
            for key in ("max_price", "min_rating"):
                try:
                    value = float(data.get(key) or 0)
                    if value > 0:
                        decision[key] = value
                except (TypeError, ValueError):
                    pass
            if str(data.get("date") or "").strip():
                decision["date"] = str(data["date"]).strip()
            if str(data.get("category") or "").strip():
                decision["category"] = str(data["category"]).strip()
            raw_list_all = str(data.get("list_all") or "").strip().lower()
            if raw_list_all in ("service", "package", "both", "none"):
                decision["list_all"] = raw_list_all
            if str(data.get("keywords") or "").strip():
                decision["keywords"] = str(data["keywords"]).strip()
        except Exception as exc:
            print(f"⚠️ 智能搜索需求解析失败，使用规则兜底: {exc}")

        # 精确名兜底：库内真实名称优先，避免“穹妹”被 LLM 当成人名
        # list_all 以 LLM 判断为主；LLM 没给出或失败时，用规则表兜底。
        if not decision.get("list_all"):
            decision["list_all"] = _list_all_kind(query) or "none"
        # 有明确分类时，优先“列出该分类”，不要变成“列出全部”
        if decision.get("category"):
            decision["list_all"] = "none"
        # 带具体特征/条件时，不能当成“列出全部”
        if decision["list_all"] != "none" and any(w in query for w in _FEATURE_WORDS):
            decision["list_all"] = "none"
        if decision["list_all"] != "none":
            decision["keywords"] = query

        exact_service = bool(fetch_services_exact(query))
        exact_housekeeper = bool(fetch_housekeepers_exact(query))
        if exact_service and not exact_housekeeper:
            decision["target"] = "service"
        elif exact_housekeeper and not exact_service:
            decision["target"] = "housekeeper"
        elif exact_service and exact_housekeeper:
            decision["target"] = "both"
        return decision

    def _call_tools(self, decision: Dict[str, Any]) -> tuple:
        results: List[Dict[str, Any]] = []
        errors: List[str] = []
        target = decision["target"]
        query = decision["keywords"] or ""
        max_price = float(decision.get("max_price") or 0)
        min_rating = float(decision.get("min_rating") or 0)
        date = decision.get("date") or ""
        category = decision.get("category") or ""

        if target in ("service", "both"):
            raw = search_services.invoke({
                "query": query,
                "max_price": max_price,
                "category": category,
                "list_all": decision.get("list_all") or "none",
            })
            parsed = self._parse_tool_result(raw)
            results.extend(parsed)
            if not parsed:
                errors.append(self._parse_tool_error(raw))

        if target in ("housekeeper", "both"):
            raw = search_housekeepers.invoke({
                "query": query,
                "min_rating": min_rating,
                "max_price": max_price,
                "date": date,
            })
            parsed = self._parse_tool_result(raw)
            results.extend(parsed)
            if not parsed:
                errors.append(self._parse_tool_error(raw))

        return results, [e for e in errors if e]

    @staticmethod
    def _parse_tool_result(raw: str) -> List[Dict[str, Any]]:
        try:
            data = json.loads(raw)
        except Exception:
            return []
        if not data.get("ok"):
            return []
        return list(data.get("data") or [])

    @staticmethod
    def _parse_tool_error(raw: str) -> str:
        try:
            data = json.loads(raw)
        except Exception:
            return ""
        if data.get("ok"):
            return ""
        return str(data.get("error") or "").strip()

    def _answer(self, query: str, results: List[Dict[str, Any]]) -> tuple:
        """返回 (自然语言回答, 最终选中的结果列表)。

        LLM 输出 selected_ids，后端只把选中的结果返回给前端，
        避免“回答说 1 个、卡片显示 2 个”的不一致。
        """
        if not results:
            return "没有找到匹配的服务或家政员，可以换个说法，或直接在家政预约页查看。", []

        compact = [
            {
                key: item.get(key)
                for key in (
                    "key", "type", "name", "price", "rating", "position",
                    "category", "available_slots", "description", "reason",
                )
                if item.get(key) not in (None, "", [])
            }
            for item in results[:8]
        ]
        try:
            response = self.llm.invoke([
                SystemMessage(content=_ANSWER_PROMPT.format(
                    results=json.dumps(compact, ensure_ascii=False),
                    query=query,
                )),
                HumanMessage(content=query),
            ])
            raw = (getattr(response, "content", "") or str(response)).strip()
            data = _parse_json(raw)
            answer = str(data.get("answer") or "").strip()
            # 小程序不渲染 Markdown，去掉常见标记
            answer = answer.replace("**", "").replace("###", "").strip()

            if answer:
                if "selected_ids" in data and isinstance(data["selected_ids"], list):
                    keys = {str(k) for k in data["selected_ids"]}
                    selected = [
                        item for item in results
                        if str(item.get("key") or "") in keys
                    ]
                    # LLM 编造了不存在的 key 时，退回原始结果，避免误隐藏
                    if keys and not selected:
                        selected = results
                    return answer, selected
                return answer, results
        except Exception as exc:
            print(f"⚠️ 智能搜索回答生成失败，使用兜底文本: {exc}")

        names = "、".join(item.get("name", "") for item in compact[:3])
        return f"为你找到：{names}。具体价格和可约时间以下单页为准。", results

    def process(self, query: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        query = (query or "").strip()
        if not query:
            return {
                "ok": False,
                "answer": "请先描述你的需求。",
                "results": [],
                "suggestions": [],
            }

        decision = self._understand(query, filters)
        results, errors = self._call_tools(decision)
        if results:
            answer, selected = self._answer(query, results)
        elif errors:
            answer = "；".join(dict.fromkeys(errors))
            selected = []
        else:
            answer = "没有找到匹配的服务或家政员，可以换个说法，或直接在家政预约页查看。"
            selected = []

        return {
            "ok": bool(selected),
            "answer": answer,
            "results": selected,
            "decision": decision,
            "suggestions": [],
        }
