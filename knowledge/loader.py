"""
① 文档加载器（Document Loader）

职责：
    - 遍历 knowledge/raw/ 下的 Markdown 文件
    - 用 UnstructuredMarkdownLoader 读取
    - 写入基础元数据：agent / category / source

不负责：
    - 文档切分（splitter.py）
    - 文本向量化（embeddings.py）
    - 向量存取（vector_store.py）

模式说明：
    - mode="single"：一个 md 文件整体返回一个 Document，切块交给 splitter.py
    - strategy="fast"：快速解析，不做复杂版面分析，适合纯文本 Markdown

依赖：
    pip install unstructured

输入输出：
    输入：无（内部从 RAW_DIR 读取）
    输出：List[Document]，一篇文件对应一个 Document
"""

from typing import List

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document

from knowledge.config import AGENT_BY_DIR, RAW_DIR


def load_raw_documents() -> List[Document]:
    """加载 raw/ 下的全部 Markdown 原文。

    Returns:
        List[Document]: 整篇文档列表，metadata 至少包含
        agent / category / source。
    """
    documents: List[Document] = []

    for path in sorted(RAW_DIR.rglob("*.md")):
        loader = UnstructuredMarkdownLoader(
            file_path=str(path),
            mode="single",
            strategy="fast",
        )
        relative = path.relative_to(RAW_DIR)

        for doc in loader.load():
            doc.metadata.update({
                # 目录名映射 Agent，如 service → service_agent
                "agent": AGENT_BY_DIR.get(relative.parts[0], "general_agent"),
                # 文件名作为知识分类，如“日常保洁”
                "category": path.stem,
                # 相对路径，供引用和增量更新使用
                "source": relative.as_posix(),
            })
            documents.append(doc)

    print(f"📄 已加载 {len(documents)} 篇文档，来源目录：{RAW_DIR}")
    return documents
