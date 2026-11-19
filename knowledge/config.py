"""
知识库全局配置。

所有可调参数集中在本文件，调参时不需要改业务代码。

计划包含：
    - 目录：RAW_DIR（知识原文）、CHROMA_DIR（向量库持久化）
    - 向量库：COLLECTION（集合名）
    - 嵌入模型：EMBEDDING_MODEL（如 BAAI/bge-m3）、超时
    - 切分参数：CHUNK_SIZE、CHUNK_OVERLAP、SEPARATORS
    - 检索参数：TOP_K、MIN_SCORE
    - 目录映射：AGENT_BY_DIR（raw 子目录名 → Agent key）
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ===== 目录 =====
PKG_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PKG_DIR.parent
RAW_DIR = PKG_DIR / "raw"

# 与项目根目录共用一份 .env（embedding 的 Key / Base URL 从这里读取）
load_dotenv(PROJECT_DIR / ".env", override=True)

# raw 子目录名 → Agent key（loader.py 使用）
AGENT_BY_DIR = {
    "service": "service_agent",
    "housekeeper": "housekeeper_agent",
    "order": "order_agent",
    "payment": "payment_agent",
    "complaint": "complaint_agent",
    "platform": "general_agent",
}

# ===== 切分参数 =====
# 知识条目多为短段落，300 字能保持一个条目/小节的完整性
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# ===== 嵌入模型（SiliconFlow，与对话模型 DeepSeek 分开配置）=====
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = os.getenv(
    "SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"
)
# 从 .env 读取模型名，缺失时补上 "openai:" 协议前缀
_model = os.getenv("SILICONFLOW_MODEL", "Pro/BAAI/bge-m3").strip()
EMBEDDING_MODEL = _model if _model.startswith("openai:") else f"openai:{_model}"

# ===== 向量库 =====
CHROMA_DIR = PROJECT_DIR / "data" / "chroma"
COLLECTION = "customer_service_kb"
TOP_K = 4
MIN_SCORE = 0.25

# ===== 检索重排（Rerank）=====
# 先用向量召回候选，再用 BGE reranker 精排，显著提升短中文条目的命中率
RERANK_ENABLED = True
RERANK_MODEL = os.getenv("SILICONFLOW_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_CANDIDATES = 120
RERANK_MIN_SCORE = 0.05

# 关键词候选与加权：弥补向量模型对短中文条目、专有词的区分度不足
KEYWORD_CANDIDATES = 20
KEYWORD_WEIGHT = 0.5
