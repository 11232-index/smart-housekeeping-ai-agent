"""
② 文档切分器（Text Splitter）

职责：
    - 把 loader 输出的整篇 Document 切成可检索的语义块
    - 优先按“一、”“3.1”这类小节标题切分，保证一个小节一个块
    - 小节过长时再用递归字符切分
    - 每块开头补标题（如【日常保洁】）
    - 继承并扩展元数据：chunk_index、chunk_id
    - 过滤过短的无效块

不负责：
    - 读取文件（loader.py）
    - 调用 embedding（embeddings.py）
    - 写入向量库（vector_store.py）

切分策略：
    1. 先按 Markdown/中文小节标题切分（标题行独立成节）
    2. 单节超过 CHUNK_SIZE 时，用 RecursiveCharacterTextSplitter 递归切分
    3. chunk_overlap 保留相邻块的重叠内容

输入输出：
    输入：List[Document]（整篇）
    输出：List[Document]（知识块）
"""

import re
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from knowledge.config import CHUNK_OVERLAP, CHUNK_SIZE

# 中文 Markdown 的分隔符优先级：段落 → 换行 → 句号 → 分号 → 逗号 → 单字符
SEPARATORS = ["\n\n", "\n", "。", "；", "，", ""]

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=SEPARATORS,
    keep_separator=True,
)

# 小节标题：一、 / 1、 / 3.1 / 第X章
_HEADING_RE = re.compile(
    r"^(?:"
    r"[一二三四五六七八九十]+、"
    r"|\d+、"
    r"|\d+\.\d+(?:\.\d+)*[、.．]?"
    r"|第[一二三四五六七八九十]+[章节]"
    r")"
)


def _split_into_sections(text: str) -> List[str]:
    """按小节标题把整篇文本切成若干节；标题行长度超过 40 字不当作标题。"""
    sections: List[str] = []
    current: List[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if current:
                current.append("")
            continue

        is_heading = bool(_HEADING_RE.match(stripped)) and len(stripped) <= 40
        if is_heading and any(item.strip() for item in current):
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)

    if any(item.strip() for item in current):
        sections.append("\n".join(current).strip())

    return [section for section in sections if section]


def split_documents(documents: List[Document]) -> List[Document]:
    """把整篇文档切分成可检索的知识块。

    Args:
        documents: loader 输出的整篇文档列表。

    Returns:
        List[Document]: 知识块列表，metadata 在原有字段基础上增加
        chunk_index / chunk_id。
    """
    chunks: List[Document] = []

    for doc in documents:
        category = doc.metadata.get("category", "")
        source = doc.metadata.get("source", "")

        kept = 0
        for section in _split_into_sections(doc.page_content):
            # 小节本身过长时，再用递归字符切分
            pieces = (
                _splitter.split_text(section)
                if len(section) > CHUNK_SIZE
                else [section]
            )

            for piece in pieces:
                body = piece.strip()
                # 过滤目录、空标题等过短内容
                if len(body) < 20:
                    continue

                chunk = Document(
                    page_content=f"【{category}】\n{body}" if category else body,
                    metadata=dict(doc.metadata),
                )
                chunk.metadata["chunk_index"] = kept
                chunk.metadata["chunk_id"] = f"{source}#{kept}"
                chunks.append(chunk)
                kept += 1

    print(f"✂️ 已切分 {len(documents)} 篇文档，共生成 {len(chunks)} 个知识块")
    return chunks
