"""把 MySQL 业务行转换成可检索的档案文本。

一条服务/套餐/家政员 = 一个 CatalogDoc：

    MySQL 一行
      ↓ extract_features() 让 LLM 推断 tags / skills / suitable_for / summary
      ↓ 拼成多行档案文本（给 embedding）
      ↓ 生成 payload（给 Qdrant 过滤 + 回表）

约定：
- 档案文本用于 embedding，决定“语义像不像”
- payload 用于精确过滤和回表，决定“是否符合条件”
- 本模块只读 MySQL，不写业务库
- LLM 只在建索引时调用，用户搜索时不会走这里
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pymysql

from smart_search.config import (
    MYSQL_DATABASE,
    MYSQL_HOST,
    MYSQL_PASSWORD,
    MYSQL_PORT,
    MYSQL_USER,
)
from smart_search.feature_extractor import extract_features

RANK_NAMES = {1: "金牌", 2: "银牌", 3: "铜牌"}


@dataclass
class CatalogDoc:
    """一条待写入 Qdrant 的目录档案。"""

    kind: str          # service / package / housekeeper
    record_id: int     # 对应 MySQL 表里的主键 id，用于回表
    text: str          # 给 embedding 的档案文本
    payload: Dict[str, Any]  # 给 Qdrant 过滤/回表用的结构化字段


def _connect():
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


def _service_doc_from_row(row: Dict[str, Any]) -> CatalogDoc:
    """服务/套餐 MySQL 行 → CatalogDoc（建库和增量同步共用）。"""
    # 第一步：让 LLM 根据名称/分类/描述推断特征。
    # 返回 tags、suitable_for、summary；结果按内容哈希缓存，
    # 这几项内容没变时不会重复调用 LLM。
    features = extract_features({
        "kind": row["kind"],
        "name": row["name"],
        "category": row["category"] or "",
        "description": row["description"] or "",
        "price": float(row["price"] or 0),
    })
    tags = features["tags"]
    suitable_for = features["suitable_for"]
    summary = features["summary"]

    # 第二步：拼成给 embedding 的档案文本。
    # 这里同时保留原始字段和 LLM 推断字段，提升自然语言召回率。
    kind_label = "服务" if row["kind"] == "service" else "套餐"
    text = "\n".join([
        f"类型：{kind_label}",
        f"名称：{row['name']}",
        f"分类：{row['category'] or '未分类'}",
        f"描述：{row['description'] or '暂无描述'}",
        f"特征：{'、'.join(tags) or '未标注'}",
        f"适合：{'、'.join(suitable_for) or '未标注'}",
        f"概括：{summary or '暂无'}",
        f"价格：{float(row['price'] or 0):.2f} 元",
    ])
    # 第三步：生成 payload，写入 Qdrant 后用于过滤和回表。
    # record_id 必须保留：检索命中后要拿它回 MySQL 查最新价格/状态。
    return CatalogDoc(
        kind=row["kind"],
        record_id=int(row["record_id"]),
        text=text,
        payload={
            "kind": row["kind"],              # 服务还是套餐
            "record_id": int(row["record_id"]),  # MySQL 主键，回表用
            "name": row["name"],              # 展示 + 精确名匹配
            "category": row["category"] or "",  # 分类过滤
            "tags": tags,                     # LLM 推断的特征标签
            "suitable_for": suitable_for,     # LLM 推断的适合人群/场景
            "summary": summary,               # LLM 一句话概括
            "price": float(row["price"] or 0),  # 预算过滤
            "status": int(row["status"] or 0),  # 上架状态过滤
        },
    )


def _service_docs() -> List[CatalogDoc]:
    """读取服务 + 套餐，生成档案列表。"""
    # 只查上架(status=1)的记录；service_item 和 service_package 用 UNION ALL 合并
    sql = """
        SELECT 'service' AS kind, s.id AS record_id, s.name, s.price, s.status,
               COALESCE(s.description, '') AS description,
               COALESCE(c.name, '') AS category
        FROM service_item s
        LEFT JOIN category c ON c.id = s.category_id
        WHERE s.status = 1
        UNION ALL
        SELECT 'package' AS kind, p.id AS record_id, p.name, p.price, p.status,
               COALESCE(p.description, '') AS description,
               COALESCE(c.name, '') AS category
        FROM service_package p
        LEFT JOIN category c ON c.id = p.category_id
        WHERE p.status = 1
        ORDER BY kind, record_id
    """
    docs: List[CatalogDoc] = []
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            for row in cur.fetchall():
                docs.append(_service_doc_from_row(row))
    return docs


def _housekeeper_doc_from_row(row: Dict[str, Any]) -> CatalogDoc:
    """家政员 MySQL 行 → CatalogDoc（建库和增量同步共用）。"""
    # 第一步：LLM 从姓名/岗位/经验/评价推断技能，例如“会做饭”“老人陪护”。
    # 家政员主要看 skills；tags/suitable_for 作为补充。
    features = extract_features({
        "kind": "housekeeper",
        "name": row["name"],
        "position": row["position"] or "",
        "past_experience": row["past_experience"] or "",
        "review": row["review"] or "",
        "rating": float(row["rating"] or 0),
        "price": float(row["price"] or 0),
        "rank": int(row["rank"] or 0),
    })
    skills = features["skills"]
    tags = features["tags"]
    suitable_for = features["suitable_for"]
    summary = features["summary"]

    # 第二步：拼成给 embedding 的文本。
    # 数字等级先翻译成“金牌/银牌/铜牌”，方便自然语言命中。
    rank_name = RANK_NAMES.get(int(row["rank"] or 0), "普通")
    text = "\n".join([
        "类型：家政员",
        f"姓名：{row['name']}",
        f"等级：{rank_name}",
        f"评分：{float(row['rating'] or 0):.1f}",
        f"单价：{float(row['price'] or 0):.2f} 元/次",
        f"岗位：{row['position'] or '未填写'}",
        f"经验：{row['past_experience'] or '未填写'}",
        f"技能：{'、'.join(skills) or '未标注'}",
        f"特征：{'、'.join(tags) or '未标注'}",
        f"适合：{'、'.join(suitable_for) or '未标注'}",
        f"概括：{summary or '暂无'}",
        f"评价：{row['review'] or '暂无评价'}",
    ])
    # 第三步：payload 里额外带上等级、评分、技能、在职状态，
    # 供 Qdrant 过滤；“无档案”的家政员 status 也会在这里体现。
    return CatalogDoc(
        kind="housekeeper",
        record_id=int(row["record_id"]),
        text=text,
        payload={
            "kind": "housekeeper",
            "record_id": int(row["record_id"]),  # MySQL 主键
            "name": row["name"],
            "rank": int(row["rank"] or 0),       # 1 金 / 2 银 / 3 铜 / 0 普通
            "rating": float(row["rating"] or 0),  # 评分过滤
            "price": float(row["price"] or 0),    # 预算过滤
            "skills": skills,                     # LLM 推断的技能
            "tags": tags,
            "suitable_for": suitable_for,
            "summary": summary,
            "status": int(row["status"] or 0),    # 员工在职状态
            "work_status": row["work_status"] or "",
        },
    )


def _housekeeper_docs() -> List[CatalogDoc]:
    """读取家政员 + 员工详情，生成档案列表。"""
    # housekeeper 本身没有技能字段，借 employee_detail 的岗位/经验补充特征
    sql = """
        SELECT h.id AS record_id, h.name, h.price, h.rating, h.`rank`,
               COALESCE(h.review, '') AS review,
               COALESCE(ed.position, '') AS position,
               COALESCE(ed.past_experience, '') AS past_experience,
               COALESCE(ed.work_status, '') AS work_status,
               COALESCE(e.status, 1) AS status
        FROM housekeeper h
        LEFT JOIN employee e ON e.id = h.employee_id
        LEFT JOIN employee_detail ed ON ed.employee_id = h.employee_id
        ORDER BY h.id
    """
    docs: List[CatalogDoc] = []
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            for row in cur.fetchall():
                docs.append(_housekeeper_doc_from_row(row))
    return docs


def build_service_doc(kind: str, record_id: int) -> Optional[CatalogDoc]:
    """按 kind + id 重建一条服务/套餐档案。

    只返回上架(status=1)的记录；下架、删除或不存在时返回 None，
    调用方据此删除 Qdrant 里的旧 point。
    """
    table = {"service": "service_item", "package": "service_package"}.get(kind)
    if not table:
        return None
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT '{kind}' AS kind, s.id AS record_id, s.name, s.price, s.status,
                       COALESCE(s.description, '') AS description,
                       COALESCE(c.name, '') AS category
                FROM {table} s
                LEFT JOIN category c ON c.id = s.category_id
                WHERE s.id = %s AND s.status = 1
                """,
                (int(record_id),),
            )
            row = cur.fetchone()
    return _service_doc_from_row(row) if row else None


def build_housekeeper_doc(record_id: int) -> Optional[CatalogDoc]:
    """按 id 重建一条家政员档案；不存在时返回 None。"""
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT h.id AS record_id, h.name, h.price, h.rating, h.`rank`,
                       COALESCE(h.review, '') AS review,
                       COALESCE(ed.position, '') AS position,
                       COALESCE(ed.past_experience, '') AS past_experience,
                       COALESCE(ed.work_status, '') AS work_status,
                       COALESCE(e.status, 1) AS status
                FROM housekeeper h
                LEFT JOIN employee e ON e.id = h.employee_id
                LEFT JOIN employee_detail ed ON ed.employee_id = h.employee_id
                WHERE h.id = %s
                """,
                (int(record_id),),
            )
            row = cur.fetchone()
    return _housekeeper_doc_from_row(row) if row else None


def housekeeper_ids_by_employee(employee_id: int) -> List[int]:
    """employee / employee_detail 变更时，找到要重建的家政员 id。"""
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM housekeeper WHERE employee_id = %s",
                (int(employee_id),),
            )
            return [int(row["id"]) for row in cur.fetchall()]


def service_refs_by_category(category_id: int) -> List[tuple]:
    """category 改名/删除时，返回该分类下要重建的 (kind, id)。"""
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 'service' AS kind, id FROM service_item
                WHERE category_id = %s AND status = 1
                UNION ALL
                SELECT 'package' AS kind, id FROM service_package
                WHERE category_id = %s AND status = 1
                """,
                (int(category_id), int(category_id)),
            )
            return [(row["kind"], int(row["id"])) for row in cur.fetchall()]


def build_all_docs() -> tuple:
    """返回 (service_docs, housekeeper_docs)。"""
    return _service_docs(), _housekeeper_docs()


if __name__ == "__main__":
    services, housekeepers = build_all_docs()
    print(f"services/packages: {len(services)}")
    for doc in services:
        print("-", doc.kind, doc.record_id, doc.payload["name"])
        print(doc.text)
    print(f"housekeepers: {len(housekeepers)}")
    for doc in housekeepers:
        print("-", doc.record_id, doc.payload["name"])
        print(doc.text)
