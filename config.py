"""
配置文件
包含系统运行所需的各种配置参数
"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# OpenAI兼容API配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "Qwen/Qwen3-8B")

# 思考模式：disabled 关闭；enabled 开启；default 使用服务商默认
OPENAI_THINKING = os.getenv("OPENAI_THINKING", "disabled").strip().lower()

# HTTP请求配置
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "30"))
HTTP_MAX_RETRIES = int(os.getenv("HTTP_MAX_RETRIES", "3"))
HTTP_HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "MultiAgentCustomerService/1.0.0"
}

# ===== MySQL（Java 后端业务库）=====
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "smart_housekeeping_system")

# 演示账号在业务库 user 表中的 id（未登录 / demo_user 时使用）
DEMO_USER_ID = int(os.getenv("DEMO_USER_ID", "2358"))

# ===== 复杂路径（Plan-and-Execute / ReAct）配置 =====
# simple/complex 分流总开关；关闭后所有请求仍走原 6 专家路径
COMPLEX_PATH_ENABLED = os.getenv("COMPLEX_PATH_ENABLED", "true").strip().lower() not in (
    "0", "false", "off", "no", "disabled"
)

# 计划与循环上限
MAX_PLAN_STEPS = int(os.getenv("MAX_PLAN_STEPS", "4"))
MAX_TOTAL_STEPS = int(os.getenv("MAX_TOTAL_STEPS", "5"))
MAX_REPLANS = int(os.getenv("MAX_REPLANS", "1"))
REACT_RECURSION_LIMIT = int(os.getenv("REACT_RECURSION_LIMIT", "8"))

# 单个步骤结果写入状态前截断，避免 checkpoint 膨胀
MAX_STEP_RESULT_CHARS = int(os.getenv("MAX_STEP_RESULT_CHARS", "2000"))

# 原生工具调用：auto=首次复杂请求探测；on=强制；off=强制确定性工具调用
REACT_NATIVE_TOOLS = os.getenv("REACT_NATIVE_TOOLS", "auto").strip().lower()

# 结构化输出的重试次数（分类 / 规划）
CLASSIFIER_JSON_RETRIES = int(os.getenv("CLASSIFIER_JSON_RETRIES", "1"))
PLANNER_JSON_RETRIES = int(os.getenv("PLANNER_JSON_RETRIES", "2"))

# LangGraph /runs/stream 读取超时（秒）
LANGGRAPH_STREAM_READ_TIMEOUT = int(os.getenv("LANGGRAPH_STREAM_READ_TIMEOUT", "300"))

# ===== 对话摘要压缩（避免静默丢弃最早消息）=====
# 开启后：对话消息超过 DIALOGUE_KEEP_MESSAGES 条时，把更早的消息压缩成结构化摘要，
# 最近 DIALOGUE_KEEP_MESSAGES 条保持原文。
DIALOGUE_COMPRESS_ENABLED = os.getenv("DIALOGUE_COMPRESS_ENABLED", "true").strip().lower() not in (
    "0", "false", "off", "no", "disabled"
)
DIALOGUE_KEEP_MESSAGES = int(os.getenv("DIALOGUE_KEEP_MESSAGES", "12"))

# 系统配置
SYSTEM_NAME = "智能家政客服系统"
VERSION = "1.0.0"

# 日志配置
LOG_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}
