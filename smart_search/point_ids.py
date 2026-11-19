"""Qdrant point id 的统一定义。

建库脚本和增量同步必须用同一套 id，否则会出现“同一条记录两个 point”
或“删除时找不到旧 point”的问题。
"""

import uuid

POINT_NAMESPACE = uuid.UUID("6f1c2a8e-3b4d-4c5e-9a7f-1d2e3f4a5b6c")


def point_id(kind: str, record_id: int) -> str:
    """kind + record_id → 稳定的 UUID。"""
    return str(uuid.uuid5(POINT_NAMESPACE, f"{kind}:{int(record_id)}"))
