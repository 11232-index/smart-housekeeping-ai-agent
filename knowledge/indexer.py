"""
离线建库编排入口。

职责：
    - 按顺序调用四个阶段：loader → splitter → vector_store
    - 决定全量重建还是按 source 增量更新
    - 提供命令行入口：
          python -m knowledge.indexer --rebuild     # 全量重建
          python -m knowledge.indexer               # 增量更新

不负责：
    - 各阶段的内部实现
    - 在线查询（retriever.py）

说明：
    Web 服务运行时不应 import 本文件，避免误触发重建。
"""

import argparse

from knowledge.loader import load_raw_documents
from knowledge.splitter import split_documents
from knowledge.vector_store import (
    add_documents,
    delete_by_source,
    reset_collection,
)


def build_index(rebuild: bool = False) -> int:
    """构建/更新知识库索引。

    Args:
        rebuild: True 清空后全量重建；False 按 source 增量更新。

    Returns:
        int: 本次写入的知识块数量。
    """
    # ① 加载原始 Markdown
    documents = load_raw_documents()

    # ② 切分成知识块
    chunks = split_documents(documents)

    # ④ 准备向量库（内部使用 ③ embedding 模型）
    if rebuild:
        reset_collection()
    else:
        sources = {
            chunk.metadata.get("source", "")
            for chunk in chunks
            if chunk.metadata.get("source")
        }
        for source in sorted(sources):
            delete_by_source(source)

    # ③→④ 向量化并写入
    add_documents(chunks)

    print(f"✅ 建库完成，共 {len(chunks)} 个知识块")
    return len(chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建智能家政客服知识库")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="清空后全量重建（首次建库或更换 embedding 模型时使用）",
    )
    args = parser.parse_args()
    build_index(rebuild=args.rebuild)
