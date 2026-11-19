"""
清理过期会话。

用法（在项目根目录）：
    python -m server.cleanup --days 90
"""

from __future__ import annotations

import argparse

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

from . import store
from .config import POSTGRES_URI, THREAD_RETENTION_DAYS


def main() -> None:
    parser = argparse.ArgumentParser(description="清理超过保留期的 LangGraph 会话")
    parser.add_argument("--days", type=int, default=THREAD_RETENTION_DAYS)
    args = parser.parse_args()

    pool = ConnectionPool(
        conninfo=POSTGRES_URI,
        min_size=1,
        max_size=5,
        open=True,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    )
    pool.wait(timeout=15)
    store.set_pool(pool)
    store.init_schema()

    saver = PostgresSaver(pool)
    saver.setup()

    count = store.cleanup_old_threads(saver, args.days)
    print(f"🧹 清理完成，共删除 {count} 个超过 {args.days} 天的会话")

    pool.close()


if __name__ == "__main__":
    main()
