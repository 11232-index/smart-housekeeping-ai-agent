"""智能搜索的 MySQL 只读查询。

Qdrant 只负责召回，最终返回给用户的名称/价格/状态/档期
都以这里查到的业务库数据为准。
"""

import datetime as _dt
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pymysql

from smart_search.config import (
    MYSQL_DATABASE,
    MYSQL_HOST,
    MYSQL_PASSWORD,
    MYSQL_PORT,
    MYSQL_USER,
)

SERVICE_TABLES = {"service": "service_item", "package": "service_package"}
REQUIREMENT_TABLES = {
    "service": ("service_requirement", "service_item_id"),
    "package": ("service_package_requirement", "service_package_id"),
}


def connect():
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
        connect_timeout=5,
    )


def _placeholders(values: Iterable[Any]) -> str:
    return ",".join(["%s"] * len(list(values)))


def _time_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, _dt.timedelta):
        total = int(value.total_seconds())
        return f"{total // 3600:02d}:{(total % 3600) // 60:02d}"
    return str(value)[:5]


def fetch_requirements(
    ids_by_kind: Dict[str, List[int]],
) -> Dict[str, List[Dict[str, str]]]:
    """批量查服务/套餐的需求项，返回 {"kind:id": [{name, value}, ...]}。

    value 与 Java 后端一致，是 JSON 数组字符串，如 '["大只","中只"]'。
    """
    result: Dict[str, List[Dict[str, str]]] = {}
    with connect() as conn:
        with conn.cursor() as cur:
            for kind, (table, column) in REQUIREMENT_TABLES.items():
                ids = ids_by_kind.get(kind) or []
                if not ids:
                    continue
                cur.execute(
                    f"""
                    SELECT {column} AS sid, name, value
                    FROM {table}
                    WHERE {column} IN ({_placeholders(ids)})
                    ORDER BY id
                    """,
                    ids,
                )
                for row in cur.fetchall():
                    result.setdefault(f"{kind}:{row['sid']}", []).append({
                        "name": row["name"] or "",
                        "value": row["value"] or "",
                    })
    return result


def _attach_requirements(rows: List[Dict[str, Any]]) -> None:
    """给查到的服务/套餐行补上 requirements 字段（原地修改）。"""
    ids_by_kind: Dict[str, List[int]] = {}
    for row in rows:
        ids_by_kind.setdefault(row["kind"], []).append(int(row["id"]))
    requirements = fetch_requirements(ids_by_kind)
    for row in rows:
        row["requirements"] = requirements.get(f'{row["kind"]}:{row["id"]}', [])


def list_all_services(kind: str = "") -> List[Dict[str, Any]]:
    """列出全部上架服务/套餐。

    kind:
        "service"  只列服务
        "package"  只列套餐
        其他/空    两者都列
    """
    wanted = {"service", "package"}
    if kind in SERVICE_TABLES:
        wanted = {kind}

    rows: List[Dict[str, Any]] = []
    with connect() as conn:
        with conn.cursor() as cur:
            for kind_name, table in SERVICE_TABLES.items():
                if kind_name not in wanted:
                    continue
                cur.execute(
                    f"""
                    SELECT s.id, s.name, s.price, s.status,
                           COALESCE(s.image, '') AS image,
                           COALESCE(s.description, '') AS description,
                           COALESCE(c.name, '') AS category
                    FROM {table} s
                    LEFT JOIN category c ON c.id = s.category_id
                    WHERE s.status = 1
                    ORDER BY s.id
                    """
                )
                for row in cur.fetchall():
                    rows.append({
                        "kind": kind_name,
                        "id": int(row["id"]),
                        "name": row["name"],
                        "price": float(row["price"] or 0),
                        "status": int(row["status"] or 0),
                        "image": row["image"] or "",
                        "description": row["description"] or "",
                        "category": row["category"] or "",
                    })
    _attach_requirements(rows)
    return rows


def fetch_categories() -> List[str]:
    """返回所有启用中的分类名。"""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT name FROM category WHERE status = 1 ORDER BY sort, id")
            return [row["name"] for row in cur.fetchall() if row["name"]]


def fetch_services_by_category(category: str) -> List[Dict[str, Any]]:
    """列出某分类下的全部上架服务/套餐。"""
    category = (category or "").strip()
    if not category:
        return []
    rows: List[Dict[str, Any]] = []
    with connect() as conn:
        with conn.cursor() as cur:
            for kind, table in SERVICE_TABLES.items():
                cur.execute(
                    f"""
                    SELECT s.id, s.name, s.price, s.status,
                           COALESCE(s.image, '') AS image,
                           COALESCE(s.description, '') AS description,
                           c.name AS category
                    FROM {table} s
                    LEFT JOIN category c ON c.id = s.category_id
                    WHERE s.status = 1
                      AND (c.name = %s OR c.name LIKE %s)
                    ORDER BY s.id
                    """,
                    (category, f"%{category}%"),
                )
                for row in cur.fetchall():
                    rows.append({
                        "kind": kind,
                        "id": int(row["id"]),
                        "name": row["name"],
                        "price": float(row["price"] or 0),
                        "status": int(row["status"] or 0),
                        "image": row["image"] or "",
                        "description": row["description"] or "",
                        "category": row["category"] or "",
                    })
    _attach_requirements(rows)
    return rows


def fetch_services_by_ids(
    ids_by_kind: Dict[str, List[int]],
) -> Dict[str, Dict[str, Any]]:
    """按 kind + id 批量查服务/套餐，返回 {"kind:id": row}。"""
    result: Dict[str, Dict[str, Any]] = {}
    with connect() as conn:
        with conn.cursor() as cur:
            for kind, table in SERVICE_TABLES.items():
                ids = ids_by_kind.get(kind) or []
                if not ids:
                    continue
                cur.execute(
                    f"""
                    SELECT s.id, s.name, s.price, s.status,
                           COALESCE(s.image, '') AS image,
                           COALESCE(s.description, '') AS description,
                           COALESCE(c.name, '') AS category
                    FROM {table} s
                    LEFT JOIN category c ON c.id = s.category_id
                    WHERE s.id IN ({_placeholders(ids)})
                    """,
                    ids,
                )
                for row in cur.fetchall():
                    result[f"{kind}:{row['id']}"] = {
                        "kind": kind,
                        "id": int(row["id"]),
                        "name": row["name"],
                        "price": float(row["price"] or 0),
                        "status": int(row["status"] or 0),
                        "image": row["image"] or "",
                        "description": row["description"] or "",
                        "category": row["category"] or "",
                    }
    _attach_requirements(list(result.values()))
    return result


def fetch_services_exact(name: str) -> List[Dict[str, Any]]:
    """按下单页展示的名称精确匹配服务/套餐（只取上架）。"""
    name = (name or "").strip()
    if not name:
        return []
    rows: List[Dict[str, Any]] = []
    with connect() as conn:
        with conn.cursor() as cur:
            for kind, table in SERVICE_TABLES.items():
                cur.execute(
                    f"""
                    SELECT s.id, s.name, s.price, s.status,
                           COALESCE(s.image, '') AS image,
                           COALESCE(s.description, '') AS description,
                           COALESCE(c.name, '') AS category
                    FROM {table} s
                    LEFT JOIN category c ON c.id = s.category_id
                    WHERE s.name = %s AND s.status = 1
                    """,
                    (name,),
                )
                for row in cur.fetchall():
                    rows.append({
                        "kind": kind,
                        "id": int(row["id"]),
                        "name": row["name"],
                        "price": float(row["price"] or 0),
                        "status": int(row["status"] or 0),
                        "image": row["image"] or "",
                        "description": row["description"] or "",
                        "category": row["category"] or "",
                    })
    _attach_requirements(rows)
    return rows


def fetch_housekeepers_by_ids(ids: List[int]) -> Dict[int, Dict[str, Any]]:
    if not ids:
        return {}
    result: Dict[int, Dict[str, Any]] = {}
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT h.id, h.name, h.price, h.rating, h.`rank`,
                       COALESCE(h.picture, '') AS picture,
                       COALESCE(h.carousel, '') AS carousel,
                       COALESCE(h.review, '') AS review,
                       COALESCE(ed.education, '') AS education,
                       COALESCE(ed.position, '') AS position,
                       COALESCE(ed.past_experience, '') AS past_experience,
                       COALESCE(ed.work_status, '') AS work_status,
                       COALESCE(ed.health_certificate, '') AS health_certificate,
                       COALESCE(e.status, 1) AS status
                FROM housekeeper h
                LEFT JOIN employee e ON e.id = h.employee_id
                LEFT JOIN employee_detail ed ON ed.employee_id = h.employee_id
                WHERE h.id IN ({_placeholders(ids)})
                """,
                ids,
            )
            for row in cur.fetchall():
                result[int(row["id"])] = {
                    "id": int(row["id"]),
                    "name": row["name"],
                    "price": float(row["price"] or 0),
                    "rating": float(row["rating"] or 0),
                    "rank": int(row["rank"] or 0),
                    "picture": row["picture"] or "",
                    "carousel": row["carousel"] or "",
                    "review": row["review"] or "",
                    "education": row["education"] or "",
                    "position": row["position"] or "",
                    "past_experience": row["past_experience"] or "",
                    "work_status": row["work_status"] or "",
                    "health_certificate": row["health_certificate"] or "",
                    "status": int(row["status"] or 0),
                }
    return result


def fetch_housekeepers_exact(name: str) -> List[Dict[str, Any]]:
    name = (name or "").strip()
    if not name:
        return []
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT h.id, h.name, h.price, h.rating, h.`rank`,
                       COALESCE(h.picture, '') AS picture,
                       COALESCE(h.carousel, '') AS carousel,
                       COALESCE(h.review, '') AS review,
                       COALESCE(ed.education, '') AS education,
                       COALESCE(ed.position, '') AS position,
                       COALESCE(ed.past_experience, '') AS past_experience,
                       COALESCE(ed.work_status, '') AS work_status,
                       COALESCE(ed.health_certificate, '') AS health_certificate,
                       COALESCE(e.status, 1) AS status
                FROM housekeeper h
                LEFT JOIN employee e ON e.id = h.employee_id
                LEFT JOIN employee_detail ed ON ed.employee_id = h.employee_id
                WHERE h.name = %s AND COALESCE(e.status, 1) = 1
                """,
                (name,),
            )
            return [
                {
                    "id": int(row["id"]),
                    "name": row["name"],
                    "price": float(row["price"] or 0),
                    "rating": float(row["rating"] or 0),
                    "rank": int(row["rank"] or 0),
                    "picture": row["picture"] or "",
                    "carousel": row["carousel"] or "",
                    "review": row["review"] or "",
                    "education": row["education"] or "",
                    "position": row["position"] or "",
                    "past_experience": row["past_experience"] or "",
                    "work_status": row["work_status"] or "",
                    "health_certificate": row["health_certificate"] or "",
                    "status": int(row["status"] or 0),
                }
                for row in cur.fetchall()
            ]


def fetch_slots(
    housekeeper_ids: List[int],
    date_from: Optional[_dt.date] = None,
    date_to: Optional[_dt.date] = None,
) -> Dict[int, List[str]]:
    """查可约档期。

    默认与小程序家政员预约页一致：明天起 3 个自然日。
    date_from/date_to 为左闭右开区间。
    """
    if not housekeeper_ids:
        return {}

    today = _dt.date.today()
    if date_from is None:
        date_from = today + _dt.timedelta(days=1)
    if date_to is None:
        date_to = date_from + _dt.timedelta(days=3)

    result: Dict[int, List[str]] = {}
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT housekeeper_id, schedule_date, start_time, end_time
                FROM housekeeper_schedule
                WHERE housekeeper_id IN ({_placeholders(housekeeper_ids)})
                  AND status = 1
                  AND schedule_date >= %s
                  AND schedule_date < %s
                ORDER BY schedule_date, start_time
                """,
                [*housekeeper_ids, date_from, date_to],
            )
            for row in cur.fetchall():
                slot = (
                    f"{row['schedule_date']} "
                    f"{_time_text(row['start_time'])}-{_time_text(row['end_time'])}"
                )
                result.setdefault(int(row["housekeeper_id"]), []).append(slot)
    return result
