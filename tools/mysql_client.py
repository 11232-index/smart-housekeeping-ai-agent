"""
MySQL 连接与查询工具。

配置来自项目根目录 config.py / .env：
    MYSQL_HOST / MYSQL_PORT / MYSQL_USER / MYSQL_PASSWORD / MYSQL_DATABASE

用法：
    from tools.mysql_client import query_all, query_one, execute, health

    rows = query_all("SELECT id, number FROM orders WHERE user_id=%s", (user_id,))
    row = query_one("SELECT COUNT(*) AS c FROM service_item")
    health()  # 打印连通性、版本、库名、表数量
"""

from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence

import pymysql
from pymysql.cursors import DictCursor

from config import (
    MYSQL_DATABASE,
    MYSQL_HOST,
    MYSQL_PASSWORD,
    MYSQL_PORT,
    MYSQL_USER,
)


def get_connection() -> pymysql.connections.Connection:
    """创建一条新的 MySQL 连接（DictCursor + autocommit）。"""
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        charset="utf8mb4",
        cursorclass=DictCursor,
        autocommit=True,
        connect_timeout=5,
    )


@contextmanager
def connection():
    """上下文管理器：自动关闭连接。"""
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()


def query_all(sql: str, params: Optional[Sequence[Any]] = None) -> List[Dict[str, Any]]:
    """执行查询，返回全部行。"""
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, params or ())
            return list(cursor.fetchall())


def query_one(sql: str, params: Optional[Sequence[Any]] = None) -> Optional[Dict[str, Any]]:
    """执行查询，返回第一行；没有则返回 None。"""
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, params or ())
            return cursor.fetchone()


def execute(sql: str, params: Optional[Sequence[Any]] = None) -> int:
    """执行写操作，返回影响行数。"""
    with connection() as conn:
        with conn.cursor() as cursor:
            return cursor.execute(sql, params or ())


def ping() -> Dict[str, Any]:
    """简单连通性检查。"""
    row = query_one("SELECT 1 AS ok, VERSION() AS version, DATABASE() AS db")
    return {"ok": bool(row and row.get("ok") == 1), **(row or {})}


def health() -> Dict[str, Any]:
    """连通性 + 版本 + 数据库名 + 表数量。"""
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT VERSION() AS version, DATABASE() AS db")
            info = cursor.fetchone() or {}
            cursor.execute("SHOW TABLES")
            rows = cursor.fetchall()
            tables = [next(iter(row.values())) for row in rows]
    return {
        "ok": True,
        **info,
        "table_count": len(tables),
        "tables": tables,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(health(), ensure_ascii=False, indent=2))
