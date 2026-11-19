"""Kafka 增量同步消费者。

启动：
    python -m smart_search_index.consumer

流程：
    Debezium topic → 解析事件 → 重建档案 → embedding → Qdrant upsert/delete

提交策略：
    只有 Qdrant 写成功（或消息被判定为无需处理）才提交 offset；
    处理失败重试后仍失败，写 DLQ 再提交，避免堵塞后续消息。
"""

import argparse
import json
import logging
import signal
import time
from typing import Optional

from confluent_kafka import Consumer, KafkaException, Producer

from smart_search.qdrant_store import ensure_all_collections
from smart_search_index.config import (
    DLQ_TOPIC,
    KAFKA_AUTO_OFFSET_RESET,
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_GROUP_ID,
    KAFKA_POLL_TIMEOUT,
    KAFKA_TOPICS,
    SYNC_MAX_RETRIES,
    SYNC_RETRY_DELAY,
)
from smart_search_index.events import ChangeEvent, parse_event
from smart_search_index.syncer import apply_event

logger = logging.getLogger("smart_search_index")

_running = True


def _stop(signum, frame) -> None:
    global _running
    _running = False
    logger.info("收到退出信号 %s，处理完当前消息后停止", signum)


def _to_dlq(producer: Producer, msg, error: Exception) -> None:
    raw = msg.value()
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            raw = repr(raw)
    payload = {
        "error": str(error),
        "topic": msg.topic(),
        "partition": msg.partition(),
        "offset": msg.offset(),
        "value": raw,
    }
    try:
        producer.produce(
            DLQ_TOPIC,
            key=msg.key(),
            value=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        )
        producer.flush(5)
        logger.warning("已写入 DLQ %s offset=%s: %s", msg.topic(), msg.offset(), error)
    except Exception as exc:  # DLQ 本身失败时至少留下日志
        logger.error("写入 DLQ 失败: %s", exc)


def _apply_with_retry(event: ChangeEvent, msg) -> Optional[Exception]:
    """返回 None 表示成功，否则返回最后一次异常。"""
    last_error: Optional[Exception] = None
    for attempt in range(1, SYNC_MAX_RETRIES + 1):
        try:
            result = apply_event(event)
            logger.info(
                "%s[%s] op=%s id=%s → %s",
                msg.topic(), msg.offset(), event.op, event.record_id, result,
            )
            return None
        except Exception as exc:
            last_error = exc
            logger.warning(
                "处理失败（第 %s/%s 次）%s id=%s: %s",
                attempt, SYNC_MAX_RETRIES, event.table, event.record_id, exc,
            )
            if attempt < SYNC_MAX_RETRIES:
                time.sleep(SYNC_RETRY_DELAY)
    return last_error


def run(group_id: str) -> None:
    ensure_all_collections()

    consumer = Consumer({
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "group.id": group_id,
        "auto.offset.reset": KAFKA_AUTO_OFFSET_RESET,
        "enable.auto.commit": False,
        # LLM 特征 + embedding 偶尔较慢，放宽单条处理上限
        "max.poll.interval.ms": 600000,
    })
    producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS})

    consumer.subscribe(KAFKA_TOPICS)
    logger.info("已订阅 %s 个 topic: %s", len(KAFKA_TOPICS), ", ".join(KAFKA_TOPICS))

    try:
        while _running:
            msg = consumer.poll(KAFKA_POLL_TIMEOUT)
            if msg is None:
                continue

            if msg.error():
                logger.error("Kafka 消息错误: %s", msg.error())
                continue

            try:
                event = parse_event(msg.value())
            except Exception as exc:
                logger.error("解析事件失败 %s offset=%s: %s", msg.topic(), msg.offset(), exc)
                _to_dlq(producer, msg, exc)
                consumer.commit(message=msg, asynchronous=False)
                continue

            if event is None:
                # tombstone / 心跳 / 未知格式：直接跳过
                consumer.commit(message=msg, asynchronous=False)
                continue

            last_error = _apply_with_retry(event, msg)
            if last_error is not None:
                _to_dlq(producer, msg, last_error)

            consumer.commit(message=msg, asynchronous=False)
    except KafkaException as exc:
        logger.error("Kafka 异常: %s", exc)
        raise
    finally:
        consumer.close()
        logger.info("消费者已退出")


def main() -> None:
    parser = argparse.ArgumentParser(description="智能搜索 Kafka 增量同步")
    parser.add_argument(
        "--group",
        default=KAFKA_GROUP_ID,
        help=f"消费者组，默认 {KAFKA_GROUP_ID}",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    # Qdrant 客户端的请求日志太吵，只保留业务事件
    logging.getLogger("httpx").setLevel(logging.WARNING)
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    run(args.group)


if __name__ == "__main__":
    main()
