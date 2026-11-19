"""智能搜索增量同步（Kafka + Debezium）。

与智能客服、智能搜索 API 完全解耦：
    MySQL binlog → Debezium → Kafka → 本包消费者 → Qdrant
"""
