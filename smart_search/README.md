# 智能搜索（独立服务）

与智能客服完全分开：独立 Agent、独立工具、独立 API、独立 Qdrant 集合。

## 目录

```
smart_search/
├── config.py            # Qdrant / embedding / LLM / MySQL 配置
├── embedding.py         # bge-m3 文本转向量
├── llm.py               # 智能搜索专用 LLM
├── feature_extractor.py # LLM 推断 tags / skills / suitable_for / summary
├── profile_builder.py   # MySQL 行 → 档案文本 + payload
├── qdrant_store.py      # Qdrant 封装
├── rerank.py            # bge-reranker 精排
├── repository.py        # MySQL 只读查询 + 档期
├── indexer.py           # 建库脚本
├── schemas.py           # API 模型
└── app.py               # FastAPI :5001
```

工具在 `tools/smart_search_tools.py`，Agent 在 `multi_agents/smart_search_agent.py`。

## 启动顺序

```bash
# 1. Qdrant（Windows 二进制）
cd E:\qdrant\qdrant-x86_64-pc-windows-msvc
qdrant.exe

# 2. 建库（内容不变会复用 LLM 特征缓存）
python -m smart_search.indexer --rebuild

# 3. 启动搜索 API
python -m uvicorn smart_search.app:app --host 0.0.0.0 --port 5001
```

健康检查：

```bash
curl http://127.0.0.1:5001/api/smart-search/health
```

## 搜索接口

```bash
curl -X POST http://127.0.0.1:5001/api/smart-search/search \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"穹妹\"}"
```

分类浏览：

```json
{"query": "春日野穹系列分类有哪些服务"}
```

会列出该分类下全部上架服务/套餐；分类不存在时明确返回“分类下暂无上架服务或套餐”。

## 小程序

入口：`pages/booking` → 搜索栏右侧「✨ 智能搜」。

`pages/smart-search/smart-search.js` 中：

```js
var SMART_SEARCH_BASE_URL = 'http://127.0.0.1:5001';
```

- 开发者工具：保持 `127.0.0.1`，并勾选“不校验合法域名”；
- 真机预览：改成电脑当前局域网 IP，例如 `http://192.168.164.119:5001`；
- 上线：通过 Nginx 暴露 HTTPS 域名，并加入小程序域名白名单。

## 说明

- 本服务不导入客服的 Agent / 路由 / 会话；
- `multi_agents/__init__.py` 和 `tools/__init__.py` 已改为惰性导入，避免互相加载；
- 自动增量同步见 `smart_search_index/README.md`（Kafka + Debezium）；
  未启动时仍可用 `indexer.py` 手动重建。
