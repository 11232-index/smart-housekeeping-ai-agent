<h1 align="center">智能家政 AI 服务端</h1>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-yellow.svg" alt="License Apache 2.0"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-%E2%89%A50.110-009688.svg" alt="FastAPI"></a>
  <a href="https://github.com/langchain-ai/langgraph"><img src="https://img.shields.io/badge/LangGraph-%E2%89%A51.0-purple.svg" alt="LangGraph"></a>
  <a href="https://www.trychroma.com/"><img src="https://img.shields.io/badge/Chroma-local-orange.svg" alt="Chroma"></a>
  <a href="https://qdrant.tech/"><img src="https://img.shields.io/badge/Qdrant-%E2%89%A51.0-red.svg" alt="Qdrant"></a>
</p>

<p align="center"><em>多智能体客服 · 语义搜索 · 知识库 · 增量索引</em></p>

## 项目概述

面向智能家政业务的 AI 后端，包含三块相对独立的能力：

1. **智能客服（5000）**：LangGraph 编排 6 个业务专家 + 1 个复杂任务 Agent，基于 MySQL 实时数据、静态知识库和多轮记忆回答用户问题。
2. **智能搜索（5001）**：独立 Agent、独立 API、独立 Qdrant 集合，负责“说不清具体名字”的语义搜索，例如“能清理宠物毛发的服务”。
3. **增量索引同步（smart_search_index）**：通过 Debezium + Kafka 监听 MySQL binlog，服务/套餐/家政员变更后自动更新 Qdrant。

## 系统架构

```
微信小程序 / Web 控制台
        │
        ├── :5000 智能客服（FastAPI）
        │     ├── LangGraph：意图分类 → 6 专家 / 复杂任务
        │     ├── tools/：MySQL 实时查询（服务、家政员、订单、优惠券……）
        │     ├── knowledge/：Chroma 静态知识库（流程、规则、FAQ）
        │     └── server/：Postgres 记忆 + Redis 缓存
        │
        └── :5001 智能搜索（FastAPI）
              ├── 精确名 / 分类浏览 / 列出全部
              ├── Qdrant 向量召回 + bge-reranker 精排
              └── MySQL 回表校验价格与状态

MySQL binlog → Debezium → Kafka → smart_search_index 消费者 → Qdrant
```

## 项目结构

```
smart-housekeeping-ai-agent/
├── multi_agents/                     # 客服智能体
│   ├── base_agent.py                 # 基类：LLM、流式输出、工具上下文、RAG
│   ├── service_agent.py              # 服务顾问
│   ├── housekeeper_agent.py          # 家政员顾问
│   ├── order_agent.py                # 预约与订单专家
│   ├── payment_agent.py              # 支付与优惠专家
│   ├── complaint_agent.py            # 投诉与售后专家
│   ├── general_agent.py              # 平台客服
│   ├── complex_agent.py              # 复杂任务：Planner / Executor / Replanner / Aggregator
│   ├── dialogue_compactor.py         # 长对话结构化摘要压缩
│   └── smart_search_agent.py         # 智能搜索 Agent（独立于客服）
│
├── tools/                            # 工具层
│   ├── business_tools.py             # 服务、家政员、订单、优惠券、估价等业务工具
│   ├── query_tools.py                # 意图分类与关键词兜底
│   ├── tool_router.py                # 简单/复杂路径的工具路由
│   ├── mysql_client.py               # MySQL 连接与查询封装
│   └── smart_search_tools.py         # 智能搜索工具：召回 → rerank → 回表
│
├── knowledge/                        # 静态知识库（RAG）
│   ├── raw/                          # Markdown 知识原文
│   │   ├── service/  order/  payment/ complaint/
│   │   ├── housekeeper/  platform/
│   ├── loader.py / splitter.py       # 文档加载、切块
│   ├── embeddings.py                 # bge-m3 向量化
│   ├── vector_store.py               # Chroma 持久化（data/chroma）
│   ├── retriever.py                  # search_knowledge / format_knowledge
│   └── indexer.py                    # 建库入口
│
├── server/                           # 智能客服 API（FastAPI）
│   ├── app.py                        # /api/chat/stream、会话、画像、控制台
│   ├── auth.py                       # JWT → user_id / openid
│   ├── store.py                      # Postgres：画像、会话元数据
│   ├── cache.py                      # Redis：画像/会话列表缓存
│   ├── cleanup.py                    # 过期会话清理
│   └── console.html                  # 客服调试控制台
│
├── smart_search/                     # 智能搜索独立服务
│   ├── app.py                        # FastAPI :5001
│   ├── repository.py                 # MySQL 只读查询
│   ├── profile_builder.py            # MySQL 行 → 检索档案
│   ├── feature_extractor.py          # LLM 推断标签/技能/摘要（带缓存）
│   ├── embedding.py / rerank.py      # bge-m3 / bge-reranker
│   ├── qdrant_store.py               # Qdrant 封装
│   ├── indexer.py                    # 全量/增量建库
│   ├── point_ids.py                  # 固定 point id（kind:id）
│   └── console.html                  # 智能搜索调试控制台
│
├── smart_search_index/               # Kafka + Debezium 增量同步
│   ├── consumer.py                   # 消费 binlog 事件，更新 Qdrant
│   ├── syncer.py                     # 事件 → 档案 → embedding → upsert/delete
│   ├── events.py                     # Debezium 事件解析
│   ├── reconcile.py                  # MySQL 与 Qdrant 对账
│   └── debezium/                     # Kafka Connect / Connector 配置模板
│
├── data/
│   ├── chroma/                       # Chroma 持久化
│   └── smart_search_features.json    # 特征抽取缓存
│
├── config.py                         # LLM、MySQL、复杂路径、对话压缩等配置
├── multi_agent_customer_service.py   # LangGraph 工作流入口
├── session_manager.py                # 会话管理器
├── langgraph.json                    # LangGraph CLI/Studio 入口配置
├── requirements.txt
├── env_example.txt
├── MEMORY_SETUP.md                   # Postgres + Redis + FastAPI 启动说明
├── smart_search/README.md            # 智能搜索服务说明
└── smart_search_index/README.md      # 增量同步部署说明
```

## 核心能力

### 1. 六个业务专家 + 综合客服

| 智能体 | 负责范围 |
|---|---|
| 服务顾问 | 服务项目、套餐、价格、服务流程 |
| 家政员顾问 | 家政员等级、评分、技能、可约时段 |
| 预约与订单专家 | 预约流程、订单状态、改约取消、催单 |
| 支付与优惠专家 | 支付异常、退款到账、优惠券规则 |
| 投诉与售后专家 | 投诉、返工补偿、物品损坏、转人工 |
| 平台客服 | 平台介绍、账号、地址、消息通知、小程序使用 |
| 综合客服 | 多意图复杂问题：Planner → Executor → Replanner → Aggregator |

- 简单问题走单专家；多意图/多步骤问题进入 Plan-and-Execute + ReAct
- Executor 优先使用原生 Function Calling，不支持时降级为确定性工具调用
- 工具失败、超时、循环次数都有护栏

### 2. 实时业务工具

服务目录、价格、家政员、档期、订单、优惠券、退款、估价等全部通过 `tools/` 查询 MySQL 业务库，不使用写死的 mock 数据。

### 3. 静态知识库（RAG）

- 目录：`knowledge/raw/` 下的 Markdown
- 向量库：本地 Chroma（`data/chroma`），集合 `customer_service_kb`
- 嵌入模型：`Pro/BAAI/bge-m3`
- 检索：按 Agent 过滤 + 相似度召回，可选 bge-reranker 精排
- 只放流程、规则、FAQ；实时服务/价格/档期仍走工具

### 4. 智能搜索（独立服务）

与智能客服完全分开：独立 Agent、独立工具、独立 API、独立 Qdrant 集合。

```
精确名 → 分类浏览 → 列出全部 → 向量召回 → rerank → MySQL 回表
```

- 支持自然语言需求，例如“适合养宠物的保洁”“清理宠物毛发”
- 结果以 MySQL 最新价格/状态为准，搜到后由小程序渲染服务/家政员卡片
- 详细说明见 [smart_search/README.md](./smart_search/README.md)

### 5. 持久化记忆

- LangGraph thread 状态存 PostgreSQL（`PostgresSaver`），重启不丢
- 用户画像、会话元数据存 PostgreSQL，按 openid 关联
- Redis 只缓存画像/会话列表，不做第二份持久化
- 长对话超过阈值后，把早期消息压缩成结构化摘要，最近消息保持原文
- 同一 `session_id` 使用 `threading.Lock` 串行执行，避免并发写 checkpoint

### 6. 流式输出

- 客服：SSE 逐 token 输出，同时发送 `answer_done` 提前通知，前端可及时恢复输入
- 复杂任务：额外推送计划与步骤进度
- 小程序：`pages/customer-service`

### 7. 增量索引同步

MySQL 的 `service_item`、`service_package`、`category`、`housekeeper`、`employee`、`employee_detail` 变更后，由 Debezium 写 binlog 事件到 Kafka，消费者自动更新 Qdrant；支持失败重试、DLQ 和全量对账。

## 快速开始

### 1. 环境要求

| 组件 | 用途 | 默认地址 |
|---|---|---|
| Python 3.10+ | 运行后端 | — |
| MySQL 8 | 业务数据 | localhost:3306 |
| PostgreSQL | LangGraph checkpoint、画像 | localhost:5432 |
| Redis | 缓存 | localhost:6379 |
| Qdrant | 智能搜索向量库 | http://127.0.0.1:6333 |
| Kafka + Debezium | 增量同步（可选） | localhost:9092 / 8083 |

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

如需使用 LangGraph Studio：

```bash
pip install langgraph-cli
pip install -U "langgraph-cli[inmem]"
```

### 3. 配置环境变量

```bash
cp env_example.txt .env
```

至少配置：

- `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL`：SiliconFlow（OpenAI 兼容接口）
- `SILICONFLOW_API_KEY`：知识库与智能搜索的 embedding / rerank
- `MYSQL_*`：Java 后端业务库
- `POSTGRES_URI`、`REDIS_URL`
- `QDRANT_URL`：默认 `http://127.0.0.1:6333`
- `USER_JWT_SECRET`：与 Java 后端 `smart.jwt.user-secret-key` 保持一致

### 4. 初始化数据库

PostgreSQL 建库、Redis 启动和 FastAPI 记忆服务步骤见 [MEMORY_SETUP.md](./MEMORY_SETUP.md)。

### 5. 构建索引

```bash
# 客服静态知识库（Chroma）
python -m knowledge.indexer --rebuild

# 智能搜索目录（Qdrant，需先启动 Qdrant）
python -m smart_search.indexer --rebuild
```

### 6. 启动服务

```bash
# 智能客服 :5000
python -m uvicorn server.app:app --host 0.0.0.0 --port 5000

# 智能搜索 :5001（可选，使用智能搜索时启动）
python -m uvicorn smart_search.app:app --host 0.0.0.0 --port 5001

# 增量同步消费者（可选，使用 Kafka/Debezium 时启动）
python -m smart_search_index.consumer
```

调试控制台：

- 客服：`http://127.0.0.1:5000/console`
- 智能搜索：`http://127.0.0.1:5001/console`
- API 文档：`http://127.0.0.1:5000/docs`、`http://127.0.0.1:5001/docs`

### 7. LangGraph Studio（开发调试）

```bash
langgraph dev
```

默认 2024 端口，通过 LangSmith Studio 查看图结构和运行轨迹。

## API 概览

### 智能客服（5000）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | Postgres / Redis 健康检查 |
| POST | `/api/chat/stream` | SSE 流式对话 |
| POST | `/api/chat` | 非流式对话（兜底） |
| POST | `/api/new_session` | 创建会话 ID |
| GET | `/api/sessions` | 会话列表 |
| GET | `/api/sessions/{thread_id}` | 会话详情 |
| DELETE | `/api/sessions/{thread_id}` | 删除会话 |
| GET / PUT | `/api/profile` | 用户画像查询/更新 |

### 智能搜索（5001）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/smart-search/health` | Qdrant 与集合健康检查 |
| POST | `/api/smart-search/search` | 语义搜索服务/套餐/家政员 |

请求示例：

```bash
curl -X POST http://127.0.0.1:5001/api/smart-search/search \
  -H "Content-Type: application/json" \
  -d '{"query":"能清理宠物毛发的服务"}'
```

## 前端对接

本仓库只包含后端。小程序与 Web 管理端是独立项目：

- 小程序：`pages/customer-service` 对接 5000；`pages/smart-search` 对接 5001
- 智能搜索地址跟随 `app.js` 的 `aiBaseUrl` 主机，端口换成 5001
- 用户 JWT 通过 `token` 请求头传给后端，密钥与 Java 后端一致

## 技术栈

| 类别 | 技术 |
|---|---|
| Web 框架 | FastAPI + Uvicorn |
| Agent 编排 | LangGraph（StateGraph、Plan-and-Execute、ReAct） |
| LLM | SiliconFlow（OpenAI 兼容接口，默认 `Qwen/Qwen3-8B`） |
| 静态知识库 | Chroma + bge-m3 + Unstructured |
| 智能搜索 | Qdrant + bge-m3 + bge-reranker-v2-m3 |
| 业务数据 | MySQL 8（PyMySQL） |
| 记忆 | PostgreSQL（LangGraph checkpoint） |
| 缓存 | Redis |
| 增量同步 | Kafka + Debezium + Kafka Connect |
| 认证 | PyJWT（HS256） |
| 流式传输 | SSE（`text/event-stream`） |

## 扩展指南

### 新增智能体

1. 在 `multi_agents/` 新建类，继承 `BaseAgent`，实现 `process`
2. 在 `multi_agents/__init__.py` 导出
3. 在 `multi_agent_customer_service.py` 注册节点和条件路由
4. 如需工具，在 `tools/business_tools.py` 的 `TOOLS_BY_AGENT` 中配置

### 新增业务工具

1. 使用 `@tool` 装饰器定义函数，写清 Google 风格 docstring
2. 在 `tools/__init__.py` 导出
3. 在 `TOOLS_BY_AGENT` 中绑定到对应 Agent

### 新增知识库文档

1. 在 `knowledge/raw/<agent目录>/` 下新增 `.md`
2. 执行 `python -m knowledge.indexer --rebuild`
3. 文档会被切块、向量化并写入 Chroma

### 新增可搜索字段

1. 修改 `smart_search/profile_builder.py` 的档案文本与 payload
2. 如涉及新表，修改 `smart_search_index/syncer.py` 和 Debezium `table.include.list`
3. 重建索引并验证对账

## 注意事项

- `knowledge/_disabled/service_catalog/` 是停用的服务目录知识，不会被加载；真实服务以 MySQL 为准
- 订单、优惠券、家政员档期不进入知识库或 Qdrant，始终通过工具实时查询
- 智能搜索新增数据后若搜不到，先执行 `python -m smart_search.indexer --rebuild`，再检查 Kafka/Debezium 是否正常
- 智能搜索依赖 Qdrant；客服主链路不依赖 Qdrant
- 增量同步与客服、搜索 API 相互独立，消费者停机不影响已有索引的查询

## 相关文档

- [MEMORY_SETUP.md](./MEMORY_SETUP.md) — PostgreSQL + Redis + FastAPI 启动
- [smart_search/README.md](./smart_search/README.md) — 智能搜索服务
- [smart_search_index/README.md](./smart_search_index/README.md) — Kafka/Debezium 增量同步
- [README_LangGraph_CLI.md](./README_LangGraph_CLI.md) — LangGraph CLI 使用
- [langgraph.json](./langgraph.json) — Studio/CLI 图入口
