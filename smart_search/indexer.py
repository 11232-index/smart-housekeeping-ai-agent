"""智能搜索目录建库脚本。

用法：
    python -m smart_search.indexer            # 增量覆盖（按 point id upsert）
    python -m smart_search.indexer --rebuild  # 清空两个集合后全量重建

只读 MySQL，写入 Qdrant，不碰智能客服代码。
"""

import argparse
from typing import Iterable, List

from qdrant_client import models

from smart_search.config import EMBED_BATCH_SIZE
from smart_search.embedding import get_embeddings
from smart_search.point_ids import point_id
from smart_search.profile_builder import CatalogDoc, build_all_docs
from smart_search.qdrant_store import (
    collection_for_kind,
    ensure_all_collections,
    ensure_collection,
    get_client,
)


def _batches(items: List[CatalogDoc], size: int) -> Iterable[List[CatalogDoc]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def index_docs(docs: List[CatalogDoc], rebuild: bool = False) -> int:
    if not docs:
        return 0

    # 服务/套餐和家政员使用不同集合；docs 里只有一种 kind
    collection = collection_for_kind(docs[0].kind)
    ensure_collection(collection, recreate=rebuild)

    embeddings = get_embeddings()
    client = get_client()
    total = 0

    for batch in _batches(docs, EMBED_BATCH_SIZE):
        # 批量把档案文本转成 1024 维向量
        vectors = embeddings.embed_documents([doc.text for doc in batch])
        # 每个档案 = 一个 point：固定 id + 向量 + payload
        points = [
            models.PointStruct(
                id=point_id(doc.kind, doc.record_id),
                vector=vector,
                payload=doc.payload,
            )
            for doc, vector in zip(batch, vectors)
        ]
        # wait=True 表示写完后才返回，便于脚本确认结果
        client.upsert(
            collection_name=collection,
            points=points,
            wait=True,
        )
        total += len(points)
        print(f"  {collection}: +{len(points)} (total {total})")

    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="智能搜索目录建库")
    parser.add_argument("--rebuild", action="store_true", help="清空集合后全量重建")
    args = parser.parse_args()

    services, housekeepers = build_all_docs()
    print(f"读取到服务/套餐 {len(services)} 条，家政员 {len(housekeepers)} 条")

    if args.rebuild:
        ensure_all_collections(recreate=True)

    n1 = index_docs(services, rebuild=args.rebuild)
    n2 = index_docs(housekeepers, rebuild=args.rebuild)

    print(f"完成：服务/套餐 {n1} 条，家政员 {n2} 条")


if __name__ == "__main__":
    main()
