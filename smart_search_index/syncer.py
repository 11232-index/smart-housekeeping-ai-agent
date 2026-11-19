"""把一条 Debezium 变更应用到 Qdrant。

原则：
- Qdrant 只存可检索档案，实时价格/状态仍由搜索接口回表 MySQL
- upsert 用固定 point id（kind + record_id），重复消费是覆盖，不是重复插入
- 记录删除 / 服务下架 → 删除 point
"""

from typing import Any, Dict

from qdrant_client import models

from smart_search.embedding import get_embeddings
from smart_search.point_ids import point_id
from smart_search.profile_builder import (
    CatalogDoc,
    build_housekeeper_doc,
    build_service_doc,
    housekeeper_ids_by_employee,
    service_refs_by_category,
)
from smart_search.qdrant_store import collection_for_kind, get_client
from smart_search_index.events import ChangeEvent


def delete_point(kind: str, record_id: int) -> None:
    get_client().delete(
        collection_name=collection_for_kind(kind),
        points_selector=models.PointIdsList(points=[point_id(kind, record_id)]),
        wait=True,
    )


def upsert_doc(doc: CatalogDoc) -> None:
    vector = get_embeddings().embed_documents([doc.text])[0]
    get_client().upsert(
        collection_name=collection_for_kind(doc.kind),
        points=[models.PointStruct(
            id=point_id(doc.kind, doc.record_id),
            vector=vector,
            payload=doc.payload,
        )],
        wait=True,
    )


def _refresh_service(kind: str, record_id: int) -> str:
    doc = build_service_doc(kind, record_id)
    if doc is None:
        delete_point(kind, record_id)
        return f"删除 {kind}:{record_id}（已下架或不存在）"
    upsert_doc(doc)
    return f"更新 {kind}:{record_id} {doc.payload.get('name', '')}"


def _refresh_housekeeper(record_id: int) -> str:
    doc = build_housekeeper_doc(record_id)
    if doc is None:
        delete_point("housekeeper", record_id)
        return f"删除 housekeeper:{record_id}（不存在）"
    upsert_doc(doc)
    return f"更新 housekeeper:{record_id} {doc.payload.get('name', '')}"


def _employee_id(event: ChangeEvent) -> int:
    """employee 用主键；employee_detail 用行里的 employee_id。"""
    if event.table == "employee":
        return event.record_id
    row: Dict[str, Any] = event.after or event.before
    try:
        return int(row.get("employee_id") or 0)
    except (TypeError, ValueError):
        return 0


def apply_event(event: ChangeEvent) -> str:
    """执行一条变更，返回可读的处理结果（用于日志）。"""
    table = event.table

    if table in ("service_item", "service_package"):
        if not event.record_id:
            raise ValueError("服务变更缺少 id")
        if event.op == "d":
            delete_point(event.kind, event.record_id)
            return f"删除 {event.kind}:{event.record_id}"
        return _refresh_service(event.kind, event.record_id)

    if table == "housekeeper":
        if not event.record_id:
            raise ValueError("家政员变更缺少 id")
        if event.op == "d":
            delete_point("housekeeper", event.record_id)
            return f"删除 housekeeper:{event.record_id}"
        return _refresh_housekeeper(event.record_id)

    if table in ("employee", "employee_detail"):
        employee_id = _employee_id(event)
        if not employee_id:
            raise ValueError("员工变更缺少 employee_id")
        housekeeper_ids = housekeeper_ids_by_employee(employee_id)
        if not housekeeper_ids:
            return f"employee:{employee_id} 没有关联家政员，跳过"
        results = [_refresh_housekeeper(hid) for hid in housekeeper_ids]
        return "；".join(results)

    if table == "category":
        if not event.record_id:
            raise ValueError("分类变更缺少 id")
        refs = service_refs_by_category(event.record_id)
        if not refs:
            return f"category:{event.record_id} 下没有上架服务，跳过"
        results = [_refresh_service(kind, rid) for kind, rid in refs]
        return "；".join(results)

    return f"忽略未监听的表 {table}"
