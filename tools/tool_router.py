"""
简单路径的工具路由器（两层）。

架构：
    1. LLM 路由器（1 次小 JSON 调用）：语义判断，输出 tool / rag / planner
    2. 执行层：按 action 分流，工具与 RAG 按需执行

不再使用关键词规则层，所有路由由 LLM 判断。规则层已移除，
避免"关键词没命中就漏掉工具"或"命中静态词就不再查工具"的死板行为。

输出统一格式：
    {
        "action": "tool" | "rag" | "planner",
        "tool": "query_order" | None,
        "args": {...},
        "need_rag": bool,
        "source": "llm" | "fallback",
    }
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from tools.business_tools import TOOLS_BY_AGENT


def _decision(
    action: str,
    tool: Optional[str] = None,
    args: Optional[Dict[str, Any]] = None,
    need_rag: bool = False,
    source: str = "llm",
) -> Dict[str, Any]:
    return {
        "action": action,
        "tool": tool,
        "args": args or {},
        "need_rag": need_rag,
        "source": source,
    }


def _dynamic_tools(agent_key: str) -> List[Any]:
    """简单路径可选的动态工具，排除 search_knowledge_base。"""
    return [
        tool for tool in TOOLS_BY_AGENT.get(agent_key, [])
        if tool.name != "search_knowledge_base"
    ]


def _tool_catalog(agent_key: str) -> str:
    tools = _dynamic_tools(agent_key)
    if not tools:
        return "（当前 Agent 没有动态工具）"
    return "\n".join(
        f"- {tool.name}({', '.join(tool.args.keys())}): "
        f"{tool.description.splitlines()[0]}"
        for tool in tools
    )


ROUTER_PROMPT = """你是智能家政客服的工具路由器。只输出 JSON，不要 Markdown，不要解释。

用户问题：{query}
当前专家：{agent_key}
当前专家可用的动态工具：
{tool_catalog}

输出四选一：
{{"action":"tool","tool":"工具名","args":{{}},"need_rag":true}}
{{"action":"tool","tool":"工具名","args":{{}},"need_rag":false}}
{{"action":"rag"}}
{{"action":"planner"}}

判断规则：
1. 问用户自己的订单、优惠券、余额、退款进度、可约档期等实时数据
   → action=tool，need_rag=false。
2. 只问具体价格数字，不需要解释规则 → action=tool，need_rag=false。
3. 问服务说明、包含哪些内容、收费规则、政策、流程、补偿、退款规则、
   注意事项、怎么/为什么/多久 → action=tool，need_rag=true。
   只要问题里有“解释、说明、包含、规则、流程、怎么、为什么、多久、注意”
   这类词，就说明还需要 RAG。
4. 只问规则、政策、流程、平台使用、FAQ → action=rag。
5. 多步骤、跨领域、需要多个工具配合 → action=planner。
6. 用户问题里出现订单号（HS 或 ORD 开头）时，填入 args.order_id。
7. 用户没有提供订单号，也没有问“我的订单/这单/订单进度/到哪一步/订单状态”时，
   不要调用 query_order。
8. 问“有哪些服务、套餐、服务项目、服务分类、服务介绍”等目录/清单问题
   → action=tool，tool=list_services，need_rag=false。
   点名了具体服务或套餐名称时 → get_service_details，是否需要解释规则由 need_rag 决定。
   问“某分类下有哪些服务”时也用 list_services，并填 args.category。
9. tool 必须是上面列出的工具之一；不需要精确参数时只填空字符串。
10. 当前专家没有动态工具时，只能返回 rag 或 planner。
11. args 只填工具参数，未知参数留空。

示例：
- “我的订单到哪一步了” → {{"action":"tool","tool":"query_order","args":{{}},"need_rag":false}}
- “你们有哪些服务/套餐” → {{"action":"tool","tool":"list_services","args":{{}},"need_rag":false}}
- “某个具体服务多少钱” → {{"action":"tool","tool":"get_service_details","args":{{"service_name":"服务名称"}},"need_rag":false}}
- “某个具体服务怎么收费，包含哪些内容” → {{"action":"tool","tool":"get_service_details","args":{{"service_name":"服务名称"}},"need_rag":true}}
- “阿姨迟到怎么办” → {{"action":"rag"}}
- “我的券能叠加吗” → {{"action":"tool","tool":"list_user_coupons","args":{{}},"need_rag":true}}

常用参数：
- query_order: order_id, user_id
- list_user_coupons: user_id, usable_only
- get_service_details: service_name
- estimate_order_price: service_name, area_sqm, hours, coupon_code
- list_services: category
- match_housekeepers: service_type, min_rating, max_price"""


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _llm_route(
    agent_key: str,
    query: str,
    state: Dict[str, Any],
    llm: Any,
) -> Optional[Dict[str, Any]]:
    messages = [
        SystemMessage(content=ROUTER_PROMPT.format(
            query=query,
            agent_key=agent_key,
            tool_catalog=_tool_catalog(agent_key),
        )),
        HumanMessage(content=f"请为这个问题选择路由：{query}"),
    ]
    try:
        response = llm.invoke(messages, temperature=0)
    except TypeError:
        response = llm.invoke(messages)

    raw = getattr(response, "content", "") or str(response)
    data = _extract_json(raw)
    if not data:
        return None

    action = str(data.get("action") or "").strip().lower()
    if action == "rag":
        return _decision("rag", need_rag=True, source="llm")
    if action == "planner":
        return _decision("planner", source="llm")
    if action != "tool":
        return None

    allowed = {tool.name for tool in _dynamic_tools(agent_key)}
    tool_name = str(data.get("tool") or "").strip()
    if tool_name not in allowed:
        return None

    raw_args = data.get("args")
    args: Dict[str, Any] = {}
    if isinstance(raw_args, dict):
        args = {
            str(key): value for key, value in raw_args.items()
            if isinstance(value, (str, int, float, bool))
        }
    if tool_name in ("query_order", "list_user_coupons"):
        args.setdefault("user_id", state.get("user_id") or "demo_user")

    need_rag = data.get("need_rag")
    if not isinstance(need_rag, bool):
        need_rag = False

    return _decision("tool", tool_name, args, need_rag=need_rag, source="llm")


def route_intent(
    agent_key: str,
    query: str,
    state: Dict[str, Any],
    llm: Any = None,
) -> Dict[str, Any]:
    """LLM 路由器直接判断；不可用时回退到只查 RAG。"""
    if llm is not None:
        try:
            decision = _llm_route(agent_key, query, state, llm)
            if decision:
                return decision
        except Exception as exc:
            print(f"⚠️ LLM 工具路由失败: {exc}")

    return _decision("rag", need_rag=True, source="fallback")
