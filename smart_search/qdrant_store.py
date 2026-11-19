"""Qdrant 访问封装。

业务代码只依赖本模块，切换本地/Cloud 只改配置。
"""

from functools import lru_cache
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient, models

from smart_search.config import (
    EMBEDDING_SIZE,
    HOUSEKEEPERS_COLLECTION,
    QDRANT_API_KEY,
    QDRANT_URL,
    SERVICES_COLLECTION,
)

PAYLOAD_INDEXES = {
    "kind": models.PayloadSchemaType.KEYWORD,
    "category": models.PayloadSchemaType.KEYWORD,
    "tags": models.PayloadSchemaType.KEYWORD,
    "skills": models.PayloadSchemaType.KEYWORD,
    "suitable_for": models.PayloadSchemaType.KEYWORD,
    "status": models.PayloadSchemaType.INTEGER,
    "record_id": models.PayloadSchemaType.INTEGER,
    "price": models.PayloadSchemaType.FLOAT,
    "rating": models.PayloadSchemaType.FLOAT,
    "rank": models.PayloadSchemaType.INTEGER,
}


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def ensure_collection(name: str, recreate: bool = False) -> None:
    client = get_client()
    if recreate and client.collection_exists(name):
        client.delete_collection(name)

    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=EMBEDDING_SIZE,
                distance=models.Distance.COSINE,
            ),
        )

    for field, schema in PAYLOAD_INDEXES.items():
        client.create_payload_index(
            collection_name=name,
            field_name=field,
            field_schema=schema,
            wait=True,
        )


def ensure_all_collections(recreate: bool = False) -> None:
    ensure_collection(SERVICES_COLLECTION, recreate=recreate)
    ensure_collection(HOUSEKEEPERS_COLLECTION, recreate=recreate)


def collection_for_kind(kind: str) -> str:
    return HOUSEKEEPERS_COLLECTION if kind == "housekeeper" else SERVICES_COLLECTION


def search(
    collection_name: str,
    query_vector: List[float],
    limit: int = 10,
    query_filter: Optional[models.Filter] = None,
    score_threshold: Optional[float] = None,
) -> List[models.ScoredPoint]:
    return get_client().query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit,
        query_filter=query_filter,
        score_threshold=score_threshold,
    ).points
