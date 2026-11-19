"""
智能家政业务工具（MySQL 版）

工具统一采用 Google 风格 docstring + @tool 装饰器：
- 普通 Python 函数承载业务逻辑
- 类型注解供框架生成参数 schema
- Args / Returns 说明供模型理解参数与返回值

数据来源：MySQL 业务库（smart_housekeeping_system），通过 tools/mysql_client.py 查询。
工具签名与 planner / ReAct 提示词保持不变。
"""

from __future__ import annotations

import datetime as _dt
import json
from decimal import Decimal
from typing import Any, Dict, List

from langchain_core.tools import BaseTool, tool

from config import DEMO_USER_ID
from tools.mysql_client import query_all, query_one

# -----------------------------------------------------------------------------
# 数据源
# -----------------------------------------------------------------------------
# 数据统一从 MySQL 业务库（smart_housekeeping_system）读取，
# 连接方式见 tools/mysql_client.py（query_all / query_one / execute / health）。
#
# 库表对应关系：
#   订单状态、金额、退款  -> orders + order_detail
#   优惠券                -> tb_user_voucher + tb_voucher
#   家政员、档期          -> housekeeper + employee_detail + housekeeper_schedule
#   服务目录、价格        -> service_item + service_package + category

# 订单状态（与 orders.status 注释一致）
ORDER_STATUS = {
    1: "待付款", 2: "待接单", 3: "已接单", 4: "服务中",
    5: "已完成", 6: "退款中", 7: "已取消",
}
# 支付状态（orders.pay_status）
PAY_STATUS = {0: "未支付", 1: "已支付", 2: "已退款"}


def _json_default(value: Any) -> Any:
    """把 MySQL 返回的 Decimal / date / datetime 转成 JSON 可序列化类型。"""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (_dt.datetime, _dt.date)):
        return value.isoformat(sep=" ") if isinstance(value, _dt.datetime) else value.isoformat()
    return str(value)


def _resolve_db_user_id(user_id: Any) -> int:
    """把会话里的 user_id（openid / user_<id> / demo_user）转成业务库数字 id。"""
    raw = str(user_id or "").strip()
    if raw.isdigit():
        return int(raw)
    if raw.startswith("user_") and raw[5:].isdigit():
        return int(raw[5:])
    if raw and raw != "demo_user":
        row = query_one("SELECT id FROM user WHERE openid = %s", (raw,))
        if row:
            return int(row["id"])
    return DEMO_USER_ID


def _ok(data: Any) -> str:
    return json.dumps({"ok": True, "data": data}, ensure_ascii=False, default=_json_default)


def _err(message: str) -> str:
    return json.dumps({"ok": False, "error": message}, ensure_ascii=False)


# -----------------------------------------------------------------------------
# 工具定义（Google 风格 docstring + @tool）
# -----------------------------------------------------------------------------

@tool
def list_services(category: str = "") -> str:
    """查询家政服务目录与参考价格。

    Args:
        category: 服务分类，可留空查询全部，或填“日常保洁”“家电清洗”“月嫂育儿”等。

    Returns:
        JSON 字符串：成功为 {"ok": true, "data": {...}}，失败为 {"ok": false, "error": "..."}。
    """
    try:
        rows = query_all(
            """
            SELECT c.name AS category, c.type AS category_type,
                   s.id AS service_id, s.name, s.price, s.description
            FROM service_item s
            LEFT JOIN category c ON c.id = s.category_id
            WHERE s.status = 1
            UNION ALL
            SELECT c.name, c.type, p.id, p.name, p.price, p.description
            FROM service_package p
            LEFT JOIN category c ON c.id = p.category_id
            WHERE p.status = 1
            """
        )
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            cat = row.get("category") or "未分类"
            if category and category not in cat and category not in (row.get("name") or ""):
                continue
            grouped.setdefault(cat, []).append({
                "name": row["name"],
                "price": float(row["price"] or 0),
                "kind": "套餐" if row.get("category_type") == 2 else "服务",
                "description": row.get("description") or "",
            })
        if not grouped:
            return _err(f"没有找到分类“{category}”" if category else "暂无服务数据")
        return _ok(grouped)
    except Exception as e:
        return _err(f"查询服务目录失败: {e}")


@tool
def get_service_details(service_name: str) -> str:
    """查询单个服务的详细内容、参考价、时长和适用场景。

    Args:
        service_name: 服务名称，例如“两室一厅日常保洁”“空调清洗”。

    Returns:
        JSON 字符串，包含分类、名称、价格、时长、服务内容和适用场景。
    """
    try:
        like = f"%{(service_name or '').strip()}%"
        row = query_one(
            """
            SELECT '服务' AS kind, s.id, s.name, s.price, s.description, c.name AS category
            FROM service_item s
            LEFT JOIN category c ON c.id = s.category_id
            WHERE s.status = 1 AND s.name LIKE %s
            ORDER BY CHAR_LENGTH(s.name)
            LIMIT 1
            """,
            (like,),
        )
        if not row:
            row = query_one(
                """
                SELECT '套餐' AS kind, p.id, p.name, p.price, p.description, c.name AS category
                FROM service_package p
                LEFT JOIN category c ON c.id = p.category_id
                WHERE p.status = 1 AND p.name LIKE %s
                ORDER BY CHAR_LENGTH(p.name)
                LIMIT 1
                """,
                (like,),
            )
        if not row:
            return _err(f"没有找到服务“{service_name}”")
        return _ok({
            "category": row.get("category") or "",
            "name": row["name"],
            "kind": row["kind"],
            "price": float(row["price"] or 0),
            "description": row.get("description") or "",
        })
    except Exception as e:
        return _err(f"查询服务详情失败: {e}")


@tool
def match_housekeepers(service_type: str = "", min_rating: float = 0.0, max_price: float = 0.0) -> str:
    """按服务类型、最低评分、最高时薪筛选家政员。

    可约时段与小程序预约页保持一致：只返回「明天起 3 个自然日」内的
    status=1 档期（明天、后天、大后天），不含今天及第 4 天以后。

    Args:
        service_type: 服务类型，例如“日常保洁”“老人陪护”，留空表示不限。
        min_rating: 最低评分，0 表示不限。
        max_price: 最高时薪，0 表示不限。

    Returns:
        JSON 字符串，包含家政员列表（等级、评分、擅长领域、可约时段）。
    """
    try:
        # 注意：housekeeper 表当前没有“擅长服务类型”字段，service_type 暂无法参与筛选
        sql = """
            SELECT h.id, h.name, h.price, h.rating, h.`rank`, h.review,
                   e.phone, ed.position, ed.work_status, ed.health_certificate
            FROM housekeeper h
            LEFT JOIN employee e ON e.id = h.employee_id
            LEFT JOIN employee_detail ed ON ed.employee_id = h.employee_id
            WHERE 1 = 1
        """
        params: List[Any] = []
        if min_rating and float(min_rating) > 0:
            sql += " AND h.rating >= %s"
            params.append(float(min_rating))
        if max_price and float(max_price) > 0:
            sql += " AND h.price <= %s"
            params.append(float(max_price))
        sql += " ORDER BY h.rating DESC, h.id"

        rows = query_all(sql, params)
        if not rows:
            return _err("没有符合条件的家政员，可放宽评分或价格条件")

        ids = [row["id"] for row in rows]
        placeholders = ",".join(["%s"] * len(ids))
        schedules = query_all(
            f"""
            SELECT housekeeper_id, schedule_date, start_time, end_time
            FROM housekeeper_schedule
            WHERE housekeeper_id IN ({placeholders})
              AND status = 1
              AND schedule_date >= CURDATE() + INTERVAL 1 DAY
              AND schedule_date <  CURDATE() + INTERVAL 4 DAY
            ORDER BY schedule_date, start_time
            """,
            ids,
        )
        slots: Dict[int, List[str]] = {}
        for item in schedules:
            slots.setdefault(item["housekeeper_id"], []).append(
                f"{item['schedule_date']} {item['start_time']}-{item['end_time']}"
            )

        result = [{
            "id": row["id"],
            "name": row["name"],
            "rating": float(row["rating"] or 0),
            "level": row.get("rank"),
            "price_per_hour": float(row["price"] or 0),
            "position": row.get("position") or "",
            "work_status": row.get("work_status"),
            "health_certificate": row.get("health_certificate"),
            "review": row.get("review") or "",
            "available_slots": slots.get(row["id"], []),
        } for row in rows]
        return _ok(result)
    except Exception as e:
        return _err(f"筛选家政员失败: {e}")


@tool
def query_order(order_id: str = "", user_id: str = "demo_user") -> str:
    """查询订单状态、服务时间、家政员、金额与退款进度。

    Args:
        order_id: 订单号，留空时返回该用户最近的订单。
        user_id: 用户 ID，默认 demo_user。

    Returns:
        JSON 字符串，包含订单详情；查不到时返回 {"ok": false, "error": "..."}。
    """
    try:
        db_user_id = _resolve_db_user_id(user_id)
        base_sql = """
            SELECT o.id, o.number, o.status, o.pay_status, o.amount,
                   o.schedule_date, o.schedule_time, o.service_duration,
                   o.address, o.contact_name, o.phone, o.remark,
                   o.cancel_reason, o.refund_reason, o.cancel_time,
                   h.name AS housekeeper_name
            FROM orders o
            LEFT JOIN housekeeper h ON h.id = o.housekeeper_id
        """
        if order_id:
            key = order_id.strip()
            order = query_one(
                base_sql + " WHERE o.number = %s OR o.id = %s LIMIT 1",
                (key, int(key) if key.isdigit() else -1),
            )
            if not order:
                return _err(f"没有找到订单 {order_id}")
        else:
            # 按下单时间返回最近订单，待付款/超时未支付/已取消都包含在内；
            # 不能用服务日期排序，否则“刚下的单”可能被服务日期更晚的旧单挤掉
            order = query_one(
                base_sql
                + """
                WHERE o.user_id = %s
                ORDER BY o.order_time DESC, o.id DESC
                LIMIT 1
                """,
                (db_user_id,),
            )
            if not order:
                return _err("该用户暂无订单")

        details = query_all(
            "SELECT name, amount FROM order_detail WHERE order_id = %s",
            (order["id"],),
        )
        result: Dict[str, Any] = {
            "order_id": order["number"],
            "status": ORDER_STATUS.get(order["status"], f"未知状态({order['status']})"),
            "pay_status": PAY_STATUS.get(order["pay_status"], ""),
            "amount": float(order["amount"] or 0),
            "scheduled_at": f"{order['schedule_date']} {order['schedule_time']}",
            "duration": order.get("service_duration"),
            "service": [
                {"name": item["name"], "amount": float(item["amount"] or 0)}
                for item in details
            ],
            "housekeeper": order.get("housekeeper_name") or "",
            "address": order.get("address") or "",
            "contact": {
                "name": order.get("contact_name") or "",
                "phone": order.get("phone") or "",
            },
            "remark": order.get("remark") or "",
        }
        if order.get("cancel_reason") or order.get("refund_reason") or order.get("cancel_time"):
            result["cancel"] = {
                "reason": order.get("cancel_reason") or "",
                "refund_reason": order.get("refund_reason") or "",
                "time": order.get("cancel_time"),
            }
        return _ok(result)
    except Exception as e:
        return _err(f"查询订单失败: {e}")


@tool
def list_user_coupons(user_id: str = "demo_user", usable_only: bool = True) -> str:
    """查询用户名下的优惠券。

    Args:
        user_id: 用户 ID，默认 demo_user。
        usable_only: 是否只返回未使用的券，默认 true。

    Returns:
        JSON 字符串，包含券码、类型、门槛、优惠力度、适用范围和有效期。
    """
    try:
        db_user_id = _resolve_db_user_id(user_id)
        sql = """
            SELECT uv.id, v.title, v.sub_title, v.rules, v.pay_value,
                   v.actual_value, v.type, uv.status, uv.claim_time, uv.use_time,
                   v.status AS voucher_status
            FROM tb_user_voucher uv
            JOIN tb_voucher v ON v.id = uv.voucher_id
            WHERE uv.user_id = %s
        """
        if usable_only:
            sql += " AND uv.status = 0"
        sql += " ORDER BY uv.claim_time DESC, uv.id DESC"

        rows = query_all(sql, (db_user_id,))
        coupons = []
        for row in rows:
            rate = float(row["actual_value"] or 0)
            coupons.append({
                "id": row["id"],
                "title": row["title"],
                "sub_title": row.get("sub_title") or "",
                "rules": row.get("rules") or "",
                "discount": f"{rate / 10:.1f} 折" if rate else "无折扣",
                "status": "未使用" if row["status"] == 0 else "已使用",
                "claimed_at": row.get("claim_time"),
                "used_at": row.get("use_time"),
                "on_sale": row.get("voucher_status") == 1,
            })
        if not coupons:
            return _ok({
                "coupons": [],
                "message": "暂无可用优惠券" if usable_only else "暂无优惠券",
            })
        return _ok({"coupons": coupons, "count": len(coupons)})
    except Exception as e:
        return _err(f"查询优惠券失败: {e}")


@tool
def estimate_order_price(service_name: str, area_sqm: int = 0, hours: int = 1, coupon_code: str = "") -> str:
    """估算订单价格并试算优惠券抵扣。

    Args:
        service_name: 服务名称，例如“两室一厅日常保洁”。
        area_sqm: 房屋面积（平方米），0 表示不按面积调整。
        hours: 钟点工服务时长（小时），默认 1。
        coupon_code: 可选优惠券码，留空表示不使用。

    Returns:
        JSON 字符串，包含基准价、优惠、抵扣金额和预估结算价。
    """
    try:
        name = (service_name or "").strip()
        if not name:
            return _err("请提供服务名称")
        like = f"%{name}%"
        base = query_one(
            """
            SELECT name, price, '服务' AS kind FROM service_item
            WHERE status = 1 AND name LIKE %s
            ORDER BY CHAR_LENGTH(name) LIMIT 1
            """,
            (like,),
        )
        if not base:
            base = query_one(
                """
                SELECT name, price, '套餐' AS kind FROM service_package
                WHERE status = 1 AND name LIKE %s
                ORDER BY CHAR_LENGTH(name) LIMIT 1
                """,
                (like,),
            )
        if not base:
            return _err(f"没有找到服务“{service_name}”")

        price = float(base["price"] or 0)
        service_title = base["name"] or ""
        if "钟点工" in service_title:
            price *= max(1, int(hours or 1))
        elif area_sqm and int(area_sqm) > 0 and "保洁" in service_title:
            price *= max(1.0, int(area_sqm) / 70.0)

        discount = 0.0
        coupon_info = ""
        if coupon_code:
            code = str(coupon_code).strip()
            coupon = query_one(
                """
                SELECT v.title, v.actual_value, uv.status
                FROM tb_user_voucher uv
                JOIN tb_voucher v ON v.id = uv.voucher_id
                WHERE uv.user_id = %s
                  AND (v.title LIKE %s OR CAST(v.id AS CHAR) = %s)
                LIMIT 1
                """,
                (DEMO_USER_ID, f"%{code}%", code),
            )
            if not coupon:
                return _err(f"没有找到优惠券 {coupon_code}")
            if coupon["status"] != 0:
                return _err(f"优惠券 {coupon_code} 已使用，不可再用")
            rate = float(coupon["actual_value"] or 0)
            if rate > 0:
                discount = price * (1 - rate / 100)
            coupon_info = f"{coupon['title']} 抵扣 {discount:.2f} 元"

        final_price = max(0.0, price - discount)
        return _ok({
            "service": service_title,
            "kind": base["kind"],
            "base_price": round(price, 2),
            "coupon": coupon_info or "未使用优惠券",
            "discount": round(discount, 2),
            "final_price": round(final_price, 2),
            "note": "实际金额以下单页结算价为准",
        })
    except Exception as e:
        return _err(f"价格估算失败: {e}")


# 售后工单工具已移除：暂不支持在线建单。
# 用户提及投诉/售后时，由 ComplaintAgent 通过 RAG 知识库回答处理规则，
# 并引导用户到订单详情提交售后或联系人工客服。


@tool
def search_knowledge_base(query: str, k: int = 4) -> str:
    """检索家政客服知识库（服务说明、订单规则、支付退款、售后政策、平台 FAQ）。

    适用于需要查静态文字资料的场景；查具体订单状态、优惠券余额、
    家政员档期等实时数据时，请使用对应的业务工具。

    Args:
        query: 要检索的问题，尽量写完整。
        k: 返回知识块数量，默认 4。

    Returns:
        JSON 字符串，包含命中的知识片段及出处。
    """
    try:
        from knowledge import format_knowledge, search_knowledge

        pairs = search_knowledge(query, agent_key=None, k=k)
        text = format_knowledge(pairs)
        if not text:
            return _err("知识库中没有找到相关内容")
        return _ok({"content": text})
    except Exception as e:
        return _err(f"知识库检索失败: {e}")


# -----------------------------------------------------------------------------
# 注册表与辅助函数
# -----------------------------------------------------------------------------

ALL_TOOLS: List[BaseTool] = [
    list_services,
    get_service_details,
    match_housekeepers,
    query_order,
    list_user_coupons,
    estimate_order_price,
    search_knowledge_base,
]

TOOL_REGISTRY: Dict[str, BaseTool] = {t.name: t for t in ALL_TOOLS}

# 按智能体限定可用工具，减少误调用
TOOLS_BY_AGENT: Dict[str, List[BaseTool]] = {
    "service_agent": [list_services, get_service_details, estimate_order_price, search_knowledge_base],
    "housekeeper_agent": [match_housekeepers, search_knowledge_base],
    "order_agent": [query_order, search_knowledge_base],
    "payment_agent": [list_user_coupons, estimate_order_price, query_order, search_knowledge_base],
    "complaint_agent": [query_order, search_knowledge_base],
    "general_agent": [search_knowledge_base],
}

# 意图 → 默认工具（planner 失败时的确定性兜底）
INTENT_DEFAULT_TOOL: Dict[str, str] = {
    "service_info": "list_services",
    "housekeeper_info": "match_housekeepers",
    "order_service": "query_order",
    "payment_voucher": "list_user_coupons",
    "complaint": "search_knowledge_base",
    "general_inquiry": "search_knowledge_base",
}

TOOL_CATALOG_FOR_PROMPT: str = "\n".join(
    f"- {t.name}({', '.join(t.args.keys())}): {t.description.splitlines()[0]}"
    for t in ALL_TOOLS
)


def run_tool_safely(name: str, args: Dict[str, Any] | None = None) -> str:
    """安全执行工具。

    Args:
        name: 工具名称。
        args: 工具参数字典，可为空。

    Returns:
        工具返回的 JSON 字符串；未知工具或执行异常时返回 {"ok": false, "error": "..."}。
    """
    tool_obj = TOOL_REGISTRY.get(name)
    if tool_obj is None:
        return _err(f"未知工具: {name}")
    try:
        # 路由器对未知参数常填空字符串；数值/布尔字段传 "" 会触发校验失败，
        # 这类空值直接丢弃，让工具使用默认值。
        clean_args: Dict[str, Any] = {}
        schema = getattr(tool_obj, "args_schema", None)
        fields = getattr(schema, "model_fields", {}) or {}
        for key, value in (args or {}).items():
            if value == "" and key in fields:
                annotation = getattr(fields[key], "annotation", None)
                if annotation in (int, float, bool):
                    continue
            clean_args[key] = value

        result = tool_obj.invoke(clean_args)
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return _err(f"工具 {name} 执行失败: {e}")


def infer_tool_for_step(title: str, query: str = "") -> str:
    """根据步骤标题/原始问题推断默认工具（确定性兜底用）。

    Args:
        title: 计划步骤标题。
        query: 用户原始问题。

    Returns:
        工具名称；无法推断时返回空字符串。
    """
    text = f"{title} {query}"

    # 静态知识：政策、规则、流程、平台使用、售后补偿等，统一走 RAG
    static_keywords = (
        "知识库", "规则", "政策", "流程", "补偿", "赔偿", "售后", "投诉",
        "爽约", "迟到", "损坏", "加价", "返工", "到账",
        "营业", "客服", "登录", "小程序", "范围", "介绍", "内容", "FAQ",
        "发票", "支付", "等级", "平台", "地址", "消息",
        "覆盖", "开通", "能服务", "可以服务", "城市", "区县", "地区",
    )
    if any(keyword in text for keyword in static_keywords):
        return "search_knowledge_base"

    # 动态数据与计算：具体订单、券、家政员、价格
    for keyword, tool_name in (
        ("优惠券", "list_user_coupons"), ("券", "list_user_coupons"),
        ("退款", "query_order"), ("订单", "query_order"), ("预约", "query_order"),
        ("家政员", "match_housekeepers"), ("阿姨", "match_housekeepers"),
        ("价格", "estimate_order_price"), ("多少钱", "estimate_order_price"),
        ("服务", "list_services"),
    ):
        if keyword in text:
            return tool_name
    return ""
