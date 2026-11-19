"""智能搜索配置。

只读取项目根目录 .env，不与智能客服的 config.py 耦合。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env", override=False)

# ===== Qdrant =====
QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip() or None
# 存有两个集合
# 服务或套餐集合
SERVICES_COLLECTION = os.getenv("QDRANT_COLLECTION_SERVICES", "smart_search_services")
# 家政员集合
HOUSEKEEPERS_COLLECTION = os.getenv(
    "QDRANT_COLLECTION_HOUSEKEEPERS", "smart_search_housekeepers"
)
EMBEDDING_SIZE = int(os.getenv("EMBEDDING_SIZE", "1024"))

# ===== Embedding（复用 SiliconFlow bge-m3）=====
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = os.getenv(
    "SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"
)
_model = os.getenv("SILICONFLOW_MODEL", "Pro/BAAI/bge-m3").strip()
EMBEDDING_MODEL = _model if _model.startswith("openai:") else f"openai:{_model}"

# ===== Rerank（bge-reranker）=====
RERANK_MODEL = os.getenv("SILICONFLOW_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_MIN_SCORE = float(os.getenv("SMART_SEARCH_RERANK_MIN_SCORE", "0.1"))
RERANK_TOP_N = int(os.getenv("SMART_SEARCH_RERANK_TOP_N", "8"))
RECALL_LIMIT = int(os.getenv("SMART_SEARCH_RECALL_LIMIT", "20"))

# ===== LLM（用于标签/技能推理）=====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "Qwen/Qwen3-8B")
LLM_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "30"))

# 特征抽取结果缓存：避免每次重建都重复调用 LLM
FEATURE_CACHE_PATH = Path(
    os.getenv("SMART_SEARCH_FEATURE_CACHE", str(PROJECT_DIR / "data" / "smart_search_features.json"))
)

# ===== MySQL（只读）=====
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "smart_housekeeping_system")

# ===== 索引参数 =====
EMBED_BATCH_SIZE = int(os.getenv("SMART_SEARCH_EMBED_BATCH_SIZE", "32"))
INDEX_BATCH_SIZE = int(os.getenv("SMART_SEARCH_INDEX_BATCH_SIZE", "64"))
