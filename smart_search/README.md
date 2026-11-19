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
├── point_ids.py         # 固定 point id（kind:id → UUID5）
├── indexer.py           # 建库脚本
├── schemas.py           # API 模型
└── app.py               # FastAPI :5001
```

工具在 `tools/smart_search_tools.py`，Agent 在 `multi_agents/smart_search_agent.py`。

## 启动顺序

```bash
# 1. Qdrant（Windows 二进制，目录按自己的安装位置）
cd <你的 Qdrant 目录>
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

页面在 `miniapp/pages/smart-search/`，地址不再写死，而是跟随 `app.js` 的 `aiBaseUrl` 自动把端口换成 5001：

```js
var _app = getApp();
var AI_BASE = (_app && _app.globalData && _app.globalData.aiBaseUrl) || 'http://127.0.0.1:5000';
var SMART_SEARCH_BASE_URL = AI_BASE.replace(/:\d+$/, ':5001');
```

- 开发者工具：`aiBaseUrl` 为 `http://127.0.0.1:5000`，自动访问 5001；需勾选“不校验合法域名”
- 真机预览：`aiBaseUrl` 改为电脑热点地址 `http://192.168.137.1:5000`，自动访问 `http://192.168.137.1:5001`；手机需连接电脑热点
- 上线：通过 Nginx 暴露 HTTPS 域名，并加入小程序域名白名单

## 说明

- 本服务不导入客服的 Agent / 路由 / 会话；
- `multi_agents/__init__.py` 和 `tools/__init__.py` 已改为惰性导入，避免互相加载；
- 自动增量同步见 `smart_search_index/README.md`（Kafka + Debezium）；
  未启动时仍可用 `indexer.py` 手动重建。
