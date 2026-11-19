"""Postgres 持久化：用户画像、会话元数据；Redis 只做缓存。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from . import cache
from .config import PROFILE_CACHE_TTL, SESSION_CACHE_TTL, THREAD_RETENTION_DAYS

_pool: Optional[ConnectionPool] = None


def set_pool(pool: ConnectionPool) -> None:
    global _pool
    _pool = pool


def get_pool() -> ConnectionPool:
    if _pool is None:
        raise RuntimeError("Postgres 连接池尚未初始化")
    return _pool


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS user_profile (
    user_id         TEXT PRIMARY KEY,
    preferred_name  TEXT,
    nickname        TEXT,
    default_address TEXT,
    preferences     JSONB NOT NULL DEFAULT '{}'::jsonb,
    last_order_id   TEXT,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS thread_meta (
    thread_id    TEXT PRIMARY KEY,
    user_id      TEXT,
    title        TEXT NOT NULL,
    last_message TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_thread_meta_user_updated
    ON thread_meta (user_id, updated_at DESC);
"""


def init_schema() -> None:
    """创建业务表（checkpointer 表由 PostgresSaver.setup() 创建）。"""
    with get_pool().connection() as conn:
        # psycopg3 的扩展协议不允许一次 execute 多条语句，逐条执行
        for statement in SCHEMA_SQL.split(";"):
            statement = statement.strip()
            if statement:
                conn.execute(statement)


def ping() -> bool:
    try:
        with get_pool().connection() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


def _iso(value: Any) -> Optional[str]:
    return value.isoformat() if isinstance(value, datetime) else (str(value) if value else None)


def _title(message: str) -> str:
    text = " ".join(str(message or "").split())
    return (text[:20] or "新会话").strip()


def _last_message(message: str) -> str:
    text = " ".join(str(message or "").split())
    return text[:60]


# ---------------------------------------------------------------------------
# 用户画像
# ---------------------------------------------------------------------------

def _row_to_profile(row) -> Optional[Dict[str, Any]]:
    if not row:
        return None
    return {
        "user_id": row[0],
        "preferred_name": row[1],
        "nickname": row[2],
        "default_address": row[3],
        "preferences": row[4] or {},
        "last_order_id": row[5],
        "updated_at": _iso(row[6]),
    }


def get_profile(user_id: Optional[str]) -> Dict[str, Any]:
    if not user_id:
        return {}

    key = f"user:profile:{user_id}"
    cached = cache.get_json(key)
    if cached is not None:
        return cached

    with get_pool().connection() as conn:
        row = conn.execute(
            """
            SELECT user_id, preferred_name, nickname, default_address,
                   preferences, last_order_id, updated_at
            FROM user_profile
            WHERE user_id = %s
            """,
            (user_id,),
        ).fetchone()

    profile = _row_to_profile(row) or {}
    cache.set_json(key, profile, PROFILE_CACHE_TTL)
    return profile


def save_profile(user_id: str, profile: Dict[str, Any]) -> None:
    if not user_id:
        return

    with get_pool().connection() as conn:
        conn.execute(
            """
            INSERT INTO user_profile
                (user_id, preferred_name, nickname, default_address,
                 preferences, last_order_id, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (user_id) DO UPDATE SET
                preferred_name  = EXCLUDED.preferred_name,
                nickname        = EXCLUDED.nickname,
                default_address = EXCLUDED.default_address,
                preferences     = EXCLUDED.preferences,
                last_order_id   = EXCLUDED.last_order_id,
                updated_at      = now()
            """,
            (
                user_id,
                profile.get("preferred_name"),
                profile.get("nickname"),
                profile.get("default_address"),
                Jsonb(profile.get("preferences") or {}),
                profile.get("last_order_id"),
            ),
        )

    cache.delete(f"user:profile:{user_id}")


# ---------------------------------------------------------------------------
# 会话元数据
# ---------------------------------------------------------------------------

def upsert_thread_meta(thread_id: str, user_id: Optional[str], message: str) -> None:
    with get_pool().connection() as conn:
        conn.execute(
            """
            INSERT INTO thread_meta (thread_id, user_id, title, last_message)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (thread_id) DO UPDATE SET
                last_message = EXCLUDED.last_message,
                updated_at   = now(),
                user_id      = COALESCE(thread_meta.user_id, EXCLUDED.user_id)
            """,
            (thread_id, user_id, _title(message), _last_message(message)),
        )
    if user_id:
        cache.delete(f"session:list:{user_id}")


def list_threads(user_id: Optional[str]) -> List[Dict[str, Any]]:
    if not user_id:
        return []

    key = f"session:list:{user_id}"
    cached = cache.get_json(key)
    if cached is not None:
        return cached

    with get_pool().connection() as conn:
        rows = conn.execute(
            """
            SELECT thread_id, title, last_message, created_at, updated_at
            FROM thread_meta
            WHERE user_id = %s
            ORDER BY updated_at DESC
            LIMIT 100
            """,
            (user_id,),
        ).fetchall()

    data = [
        {
            "session_id": row[0],
            "thread_id": row[0],
            "title": row[1],
            "last_message": row[2],
            "created_at": _iso(row[3]),
            "updated_at": _iso(row[4]),
        }
        for row in rows
    ]
    cache.set_json(key, data, SESSION_CACHE_TTL)
    return data


def get_thread_meta(thread_id: str) -> Optional[Dict[str, Any]]:
    with get_pool().connection() as conn:
        row = conn.execute(
            """
            SELECT thread_id, user_id, title, last_message, created_at, updated_at
            FROM thread_meta
            WHERE thread_id = %s
            """,
            (thread_id,),
        ).fetchone()

    if not row:
        return None
    return {
        "session_id": row[0],
        "thread_id": row[0],
        "user_id": row[1],
        "title": row[2],
        "last_message": row[3],
        "created_at": _iso(row[4]),
        "updated_at": _iso(row[5]),
    }


def delete_thread_meta(thread_id: str) -> Optional[str]:
    """删除会话元数据，返回它所属的 user_id。"""
    meta = get_thread_meta(thread_id)
    if not meta:
        return None

    with get_pool().connection() as conn:
        conn.execute("DELETE FROM thread_meta WHERE thread_id = %s", (thread_id,))

    user_id = meta.get("user_id")
    if user_id:
        cache.delete(f"session:list:{user_id}")
    return user_id


# ---------------------------------------------------------------------------
# 清理
# ---------------------------------------------------------------------------

def cleanup_old_threads(saver, days: int = THREAD_RETENTION_DAYS) -> int:
    """删除超过保留期的 thread（checkpointer + 元数据）。返回删除数量。"""
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, days))

    with get_pool().connection() as conn:
        rows = conn.execute(
            "SELECT thread_id FROM thread_meta WHERE updated_at < %s",
            (cutoff,),
        ).fetchall()

    count = 0
    for (thread_id,) in rows:
        user_id = delete_thread_meta(thread_id)
        try:
            saver.delete_thread(thread_id)
        except Exception as e:
            print(f"⚠️ saver.delete_thread 失败 {thread_id}: {e}")
            # 兜底 SQL：不同版本表名可能略有差异
            with get_pool().connection() as conn:
                for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
                    try:
                        conn.execute(f"DELETE FROM {table} WHERE thread_id = %s", (thread_id,))
                    except Exception:
                        pass
        if user_id:
            print(f"🧹 已清理会话 {thread_id} (user={user_id})")
        count += 1

    return count
