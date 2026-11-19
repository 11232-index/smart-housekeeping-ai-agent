"""增量同步配置（只读 .env）。"""

import os

from smart_search.config import MYSQL_DATABASE

# ===== Kafka =====
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:9092")
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "smart-search-indexer")
KAFKA_AUTO_OFFSET_RESET = os.getenv("KAFKA_AUTO_OFFSET_RESET", "earliest")
KAFKA_POLL_TIMEOUT = float(os.getenv("KAFKA_POLL_TIMEOUT", "1.0"))

# Debezium topic 命名：<topic.prefix>.<database>.<table>
DEBEZIUM_TOPIC_PREFIX = os.getenv("DEBEZIUM_TOPIC_PREFIX", "housekeeping")

WATCHED_TABLES = (
    "service_item",
    "service_package",
    "category",
    "housekeeper",
    "employee",
    "employee_detail",
)

KAFKA_TOPICS = [
    f"{DEBEZIUM_TOPIC_PREFIX}.{MYSQL_DATABASE}.{table}"
    for table in WATCHED_TABLES
]

# 失败消息进入的死信队列
DLQ_TOPIC = os.getenv("KAFKA_DLQ_TOPIC", "smart_search.dlq")

# ===== 重试 =====
SYNC_MAX_RETRIES = int(os.getenv("SMART_SEARCH_SYNC_RETRIES", "3"))
SYNC_RETRY_DELAY = float(os.getenv("SMART_SEARCH_SYNC_RETRY_DELAY", "2.0"))
