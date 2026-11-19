"""Debezium 变更事件解析。

兼容两种格式：
    1. value.converter.schemas.enable=false（推荐）
       {"op":"u","before":{...},"after":{...},"source":{...}}
    2. schemas.enable=true
       {"schema":{...},"payload":{"op":"u","before":{...},"after":{...}}}
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

# 服务目录两份表 → Qdrant 里的 kind
TABLE_KIND = {
    "service_item": "service",
    "service_package": "package",
}

VALID_OPS = {"r", "c", "u", "d"}


@dataclass
class ChangeEvent:
    op: str
    table: str
    record_id: int
    before: Dict[str, Any]
    after: Dict[str, Any]

    @property
    def kind(self) -> str:
        return TABLE_KIND.get(self.table, "")


def _unwrap(value: Any) -> Optional[Dict[str, Any]]:
    """去掉 Kafka Connect 的 schema 外壳。"""
    if not isinstance(value, dict):
        return None
    payload = value.get("payload")
    if isinstance(payload, dict) and "op" in payload:
        return payload
    return value


def parse_event(raw: Any) -> Optional[ChangeEvent]:
    """Kafka 消息 value → ChangeEvent。

    无法识别（tombstone、心跳、未知表）时返回 None。
    """
    if raw is None:
        return None

    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None

    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            return None

    value = _unwrap(raw)
    if not value:
        return None

    op = str(value.get("op") or "").lower()
    if op not in VALID_OPS:
        return None

    source = value.get("source") or {}
    table = str(source.get("table") or "").strip()
    if not table:
        return None

    before = value.get("before") or {}
    after = value.get("after") or {}
    row = after or before
    try:
        record_id = int(row.get("id") or 0)
    except (TypeError, ValueError):
        record_id = 0

    return ChangeEvent(
        op=op,
        table=table,
        record_id=record_id,
        before=before if isinstance(before, dict) else {},
        after=after if isinstance(after, dict) else {},
    )
