"""MySQL 与 Qdrant 对账。

用法：
    python -m smart_search_index.reconcile          # 只检查，不修改
    python -m smart_search_index.reconcile --fix    # 补建缺失、删除多余

Kafka 漏消息、消费者长时间停机、手动改库都可能造成两边不一致，
定时跑一次对账即可兜底。
"""

import argparse
from typing import Dict, List, Set

from qdrant_client import models

from smart_search.config import HOUSEKEEPERS_COLLECTION, SERVICES_COLLECTION
from smart_search.indexer import index_docs
from smart_search.point_ids import point_id
from smart_search.profile_builder import CatalogDoc, build_all_docs
from smart_search.qdrant_store import get_client


def _actual_ids(collection: str) -> Set[str]:
    client = get_client()
    ids: Set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        ids.update(str(point.id) for point in points)
        if offset is None:
            break
    return ids


def _expected(docs: List[CatalogDoc]) -> Dict[str, CatalogDoc]:
    return {point_id(doc.kind, doc.record_id): doc for doc in docs}


def _check(label: str, collection: str, docs: List[CatalogDoc], fix: bool) -> None:
    expected = _expected(docs)
    actual = _actual_ids(collection)

    missing_ids = sorted(set(expected) - actual)
    extra_ids = sorted(actual - set(expected))
    print(f"\n{label}: MySQL {len(expected)} 条，Qdrant {len(actual)} 条")
    print(f"  缺失 {len(missing_ids)} 条，多余 {len(extra_ids)} 条")

    if not fix:
        if missing_ids:
            for pid in missing_ids[:5]:
                print(f"    缺失示例: {expected[pid].kind}:{expected[pid].record_id}")
        if extra_ids:
            for pid in extra_ids[:5]:
                print(f"    多余示例 point_id: {pid}")
        return

    if missing_ids:
        missing_docs = [expected[pid] for pid in missing_ids]
        n = index_docs(missing_docs)
        print(f"  已补建 {n} 条")

    if extra_ids:
        get_client().delete(
            collection_name=collection,
            points_selector=models.PointIdsList(points=extra_ids),
            wait=True,
        )
        print(f"  已删除 {len(extra_ids)} 条多余 point")


def main() -> None:
    parser = argparse.ArgumentParser(description="智能搜索索引对账")
    parser.add_argument("--fix", action="store_true", help="补建缺失并删除多余")
    args = parser.parse_args()

    services, housekeepers = build_all_docs()
    _check("服务/套餐", SERVICES_COLLECTION, services, args.fix)
    _check("家政员", HOUSEKEEPERS_COLLECTION, housekeepers, args.fix)
    print("\n对账完成" + ("（已修复）" if args.fix else "（仅检查）"))


if __name__ == "__main__":
    main()
