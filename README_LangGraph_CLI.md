# 使用 LangGraph CLI / Studio 调试

项目的主运行方式是 FastAPI（`server.app:app`，端口 5000）。LangGraph CLI 用于本地图结构调试、运行轨迹查看和 Studio 可视化，不是小程序生产环境的必需组件。

## 1. 当前 langgraph.json

```json
{
  "dependencies": ["."],
  "graphs": {
    "customer_service": "./multi_agent_customer_service.py:make_graph"
  },
  "env": "./.env",
  "python_version": "3.11",
  "base_image": "langchain/langgraph-api:0.2",
  "store": {
    "ttl": {
      "refresh_on_read": true,
      "sweep_interval_minutes": 60,
      "default_ttl": 10080
    }
  },
  "checkpointer": {
    "ttl": {
      "strategy": "delete",
      "sweep_interval_minutes": 10,
      "default_ttl": 43200
    }
  }
}
```

说明：

- `graphs.customer_service` 指向 `multi_agent_customer_service.py` 的 `make_graph`
- 工作流节点和条件路由都写在 Python 里，`langgraph.json` 只负责声明入口
- `python_version` 是 LangGraph 平台镜像使用的版本，本机运行不受它限制
- `store` / `checkpointer` 是 LangGraph 平台的 TTL 配置；小程序使用的记忆由 FastAPI + `PostgresSaver` 负责

## 2. 安装

```bash
pip install langgraph-cli
pip install -U "langgraph-cli[inmem]"
```

Windows 下如果 `langgraph` 不在 PATH，使用完整路径，例如：

```text
C:\Users\<你的用户名>\AppData\Roaming\Python\Python312\Scripts\langgraph.exe
```

验证：

```bash
langgraph --help
```

## 3. 启动开发服务

```bash
langgraph dev
```

- 默认端口：`2024`
- 会读取 `langgraph.json` 和 `.env`
- 支持热重载，修改代码后自动重启

常用参数：

```bash
# 指定配置
langgraph dev -c langgraph.json

# 指定端口和主机
langgraph dev --port 2024 --host 0.0.0.0

# 断点调试
langgraph dev --debug-port 5678 --wait-for-client
```

## 4. LangGraph Studio

启动 `langgraph dev` 后，打开 Studio：

```text
https://smith.langchain.com/studio/thread?render=interact&baseUrl=http://127.0.0.1:2024
```

可查看：

- 图结构与节点流转
- 每个节点的输入/输出
- 工具调用与 token 流
- thread 历史与 checkpoint

## 5. HTTP API

LangGraph 开发服务默认提供：

```text
http://127.0.0.1:2024/docs
http://127.0.0.1:2024/ok
```

常用端点：

| 分类 | 端点 |
|---|---|
| Assistants | `POST /assistants`、`POST /assistants/search`、`GET/PATCH/DELETE /assistants/{id}` |
| Threads | `POST /threads`、`POST /threads/search`、`GET/PATCH/DELETE /threads/{id}` |
| Runs | `POST /threads/{thread_id}/runs`、`POST /threads/{thread_id}/runs/stream` |
| 运行状态 | `GET /threads/{thread_id}/runs/{run_id}`、`POST .../cancel` |
| 无状态 | `POST /runs`、`POST /runs/stream` |
| 系统 | `GET /info`、`GET /ok`、`GET /metrics` |

小程序实际调用的是项目自定义 FastAPI：

```text
POST http://127.0.0.1:5000/api/chat/stream
```

两者不要混用端口。

## 6. 环境变量

`langgraph dev` 会读取项目根目录 `.env`，至少需要：

```env
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
OPENAI_MODEL=Qwen/Qwen3-8B

MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_mysql_password
MYSQL_DATABASE=smart_housekeeping_system
```

完整配置见 `env_example.txt`。`.env` 不要提交到 Git。

## 7. Docker（可选）

需要本机安装 Docker：

```bash
# 构建镜像
langgraph build -t smart-housekeeping-ai:latest

# 启动服务
langgraph up --image smart-housekeeping-ai:latest

# 生成 Dockerfile
langgraph dockerfile -c langgraph.json Dockerfile
```

注意：容器内访问 `localhost` 指向容器自身，连接 MySQL/Postgres 时需要改成宿主机地址或容器网络地址。

## 8. 常见问题

| 现象 | 处理 |
|---|---|
| `langgraph` 不是命令 | 把 Scripts 目录加入 PATH，或用完整路径 |
| 启动时报 `.env` 编码错误 | 把 `.env` 另存为 UTF-8，注释尽量使用英文 |
| 2024 端口被占用 | `langgraph dev --port <其他端口>` |
| Studio 打不开 | 确认 `langgraph dev` 已启动，并允许访问 `127.0.0.1:2024` |
| 图入口找不到 | 确认 `langgraph.json` 中路径是 `./multi_agent_customer_service.py:make_graph` |
