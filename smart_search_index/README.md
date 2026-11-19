# 智能搜索增量同步（Kafka + Debezium）

MySQL 一变更，Qdrant 索引自动更新，不再手动跑 `smart_search.indexer`。

```
MySQL(binlog) → Debezium → Kafka → consumer.py → Qdrant
```

## 1. MySQL 准备

确认开启 binlog（`my.ini`）：

```ini
[mysqld]
log-bin=mysql-bin
binlog_format=ROW
binlog_row_image=FULL
server-id=1
```

重建账号：

```sql
CREATE USER 'debezium'@'%' IDENTIFIED BY 'debezium';
GRANT SELECT, RELOAD, SHOW DATABASES, REPLICATION SLAVE, REPLICATION CLIENT
  ON *.* TO 'debezium'@'%';
FLUSH PRIVILEGES;
```

## 2. 启动 Kafka

```powershell
# KRaft 模式（Kafka 3.3+）
cd C:\kafka
bin\windows\kafka-storage.bat random-uuid        # 首次
bin\windows\kafka-storage.bat format -t <uuid> -c config\kraft\server.properties
bin\windows\kafka-server-start.bat config\kraft\server.properties
```

Windows 内存有限时在 `server.properties` 或启动脚本里设：

```
-Xms256m -Xmx512m
```

## 3. 启动 Debezium

1. 下载 Debezium MySQL connector，解压到 `C:\debezium\plugin`
2. 复制本目录 `debezium/*.properties` 到 Kafka 的 `config\`
3. 按实际路径修改 `plugin.path`、`offset.storage.file.filename`
4. 确认 `topic.prefix=housekeeping` 与 `smart_search_index/config.py` 的
   `DEBEZIUM_TOPIC_PREFIX` 一致

```powershell
cd C:\kafka
bin\windows\connect-standalone.bat config\connect-standalone.properties config\connect-mysql-source.properties
```

检查 connector：

```powershell
curl http://127.0.0.1:8083/connectors/smart-housekeeping-connector/status
```

首次启动会做一次全量 snapshot，之后走增量。

## 4. 启动消费者

```powershell
cd "D:\桌面\smart-housekeeping-ai-agent"
python -m smart_search_index.consumer
```

日志里出现 `已订阅 N 个 topic` 即正常。

## 5. 验证

```sql
UPDATE service_item SET price = 99 WHERE id = 90;
```

看消费者日志是否打印 `更新 service:90 ...`，然后：

```powershell
curl -X POST http://127.0.0.1:5001/api/smart-search/search `
  -H "Content-Type: application/json" `
  -d "{\"query\":\"韩湘子\"}"
```

价格应为 99。

## 6. 对账

Kafka 停太久、消费者漏消息时：

```powershell
python -m smart_search_index.reconcile        # 只检查
python -m smart_search_index.reconcile --fix  # 补建缺失、删除多余
```

建议每小时或每天跑一次。

## 7. 监听范围

| MySQL 表 | 处理方式 |
|---|---|
| `service_item` / `service_package` | 重建服务/套餐 point；下架或删除则删 point |
| `category` | 重建该分类下全部服务/套餐 |
| `housekeeper` | 重建家政员 point；删除则删 point |
| `employee` / `employee_detail` | 找到关联家政员后重建 |

实时订单、优惠券、档期不进索引，搜索时仍回表 MySQL。

## 8. 失败处理

- 单条处理失败重试 3 次（间隔 2 秒）
- 仍失败写入 `smart_search.dlq`，日志保留原始消息
- 消息处理成功后才提交 offset，重启不会丢

## 9. 注意

- Qdrant point id 固定为 `UUID5(kind:record_id)`，建库和增量同步共用；
  从旧版本升级后建议先跑一次 `python -m smart_search.indexer --rebuild`
- 特征抽取结果按内容哈希缓存，只改价格不会重复调用 LLM
- Kafka / Debezium 版本差异主要在 `topic.prefix`（Debezium 1.x 是 `database.server.name`）
