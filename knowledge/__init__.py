"""
静态 RAG 知识库包。

对外只暴露两个函数，业务层（Agent / 工具）不需要感知 Chroma、Embedding 等实现细节：
    - search_knowledge: 在线检索，返回 [(Document, score), ...]
    - format_knowledge: 把检索结果格式化成可拼进 Prompt 的文本

包内文件分工：
    config.py        全局配置（目录、模型、chunk 参数、检索参数）
    loader.py        ① 文档加载器：读 raw/*.md，补齐 agent / category / source 元数据
    splitter.py      ② 文档切分器：切块、补标题、生成 chunk_id
    embeddings.py    ③ 文档嵌入模型：文本 → 向量
    vector_store.py  ④ 向量存储：Chroma 建库、增删、相似度检索
    indexer.py       离线建库编排入口（python -m knowledge.indexer）
    retriever.py     在线检索入口（Agent 和工具只调用它）
"""

from .retriever import format_knowledge, search_knowledge

__all__ = ["search_knowledge", "format_knowledge"]
