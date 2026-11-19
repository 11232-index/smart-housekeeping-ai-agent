"""记忆服务配置（从项目根目录 .env 读取）。"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Postgres：LangGraph checkpointer + 用户画像/会话元数据
POSTGRES_URI = os.getenv(
    "POSTGRES_URI",
    "postgresql://langgraph:langgraph@localhost:5432/langgraph",
)

# Redis：只做缓存，不做持久化
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
PROFILE_CACHE_TTL = int(os.getenv("PROFILE_CACHE_TTL", "1800"))
SESSION_CACHE_TTL = int(os.getenv("SESSION_CACHE_TTL", "60"))

# 与 Java application.yml 的 smart.jwt.user-secret-key 保持一致
USER_JWT_SECRET = os.getenv("USER_JWT_SECRET", "change-me-secret")

# 会话保留天数（清理脚本使用）
THREAD_RETENTION_DAYS = int(os.getenv("THREAD_RETENTION_DAYS", "90"))
