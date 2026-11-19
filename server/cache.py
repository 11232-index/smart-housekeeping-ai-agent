"""Redis 缓存封装：只缓存可从 Postgres 重建的数据，Redis 不可用时自动降级。"""

from __future__ import annotations

import json
from typing import Any, Optional

import redis

from .config import REDIS_URL

_client: Optional[redis.Redis] = None
_checked = False


def client() -> Optional[redis.Redis]:
    global _client, _checked
    if _checked:
        return _client

    _checked = True
    try:
        c = redis.Redis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=1,
            socket_timeout=1,
            # 兼容 Redis 3.2：强制 RESP2，避免发送 Redis 6+ 才支持的 HELLO
            protocol=2,
        )
        c.ping()
        _client = c
        print("✅ Redis 已连接，画像/会话列表缓存已启用")
    except Exception as e:
        _client = None
        print(f"⚠️ Redis 不可用，缓存降级为直接查 Postgres: {e}")
    return _client


def get_json(key: str) -> Optional[Any]:
    c = client()
    if c is None:
        return None
    try:
        raw = c.get(key)
        return json.loads(raw) if raw else None
    except Exception as e:
        print(f"⚠️ Redis 读取失败 {key}: {e}")
        return None


def set_json(key: str, value: Any, ttl: int) -> None:
    c = client()
    if c is None:
        return
    try:
        c.setex(key, ttl, json.dumps(value, ensure_ascii=False))
    except Exception as e:
        print(f"⚠️ Redis 写入失败 {key}: {e}")


def delete(key: str) -> None:
    c = client()
    if c is None:
        return
    try:
        c.delete(key)
    except Exception as e:
        print(f"⚠️ Redis 删除失败 {key}: {e}")


def ping() -> bool:
    c = client()
    if c is None:
        return False
    try:
        return bool(c.ping())
    except Exception:
        return False
