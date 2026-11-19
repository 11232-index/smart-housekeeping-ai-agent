# 记忆服务启动说明（FastAPI + PostgreSQL + Redis）

本方案用 **PostgresSaver** 保存 LangGraph 的 thread 状态，用 **Redis** 缓存用户画像/会话列表。
不再需要 `langgraph dev`；FastAPI 直接占用 5000，小程序无需改地址。

## 0. 目录与端口

| 服务 | 端口 | 说明 |
|---|---|---|
| PostgreSQL | 5432 | checkpointer + user_profile + thread_meta |
| Redis | 6379 | 画像/会话列表缓存 |
| Java 后端 | 8080 | 微信登录、业务接口 |
| FastAPI 记忆服务 | 5000 | 客服 API + 记忆服务（小程序对接） |
| 小程序开发者工具 | - | `aiBaseUrl` 仍为 `http://127.0.0.1:5000` |

## 1. 创建 Postgres 数据库和用户

`psql` 不在 PATH 时用全路径：

```powershell
& "E:\PostgreSQL\bin\psql.exe" -h localhost -p 5432 -U postgres
```

输入安装时设置的 postgres 密码，然后执行：

```sql
CREATE USER langgraph WITH PASSWORD 'langgraph';
CREATE DATABASE langgraph OWNER langgraph;
```

如果密码不是 `langgraph`，同步修改 `.env` 里的 `POSTGRES_URI`。

验证：

```powershell
& "E:\PostgreSQL\bin\psql.exe" -h localhost -p 5432 -U langgraph -d langgraph
```

## 2. 启动 Redis

```powershell
cd "D:\桌面\项目\资料\资料\day05\Redis-x64-3.2.100"
.\redis-server.exe redis.windows.conf
```

另开窗口验证：

```powershell
cd "D:\桌面\项目\资料\资料\day05\Redis-x64-3.2.100"
.\redis-cli.exe ping
```

返回 `PONG` 即可。

> 代码里已设置 `protocol=2`，兼容 Redis 3.2；否则 redis-py 8 会发送 `HELLO` 导致连接失败。

## 3. 安装 Python 依赖

```powershell
cd "D:\桌面\smart-housekeeping-ai-agent"
pip install -r requirements.txt
```

关键包：

```bash
pip install "langgraph-checkpoint-postgres==3.0.5" "psycopg[binary]" psycopg-pool redis fastapi uvicorn pyjwt
```

## 4. 启动 Java 后端（可选，但登录需要）

用 IntelliJ 运行 `SmartApplication`，或：

```powershell
cd "D:\桌面\后端初始工程\housekeeping-digital-platform"
mvn -pl smart-server -am spring-boot:run
```

Java 侧已完成：

- `/user/user/login` 用 code 换 openid
- JWT claims 增加 `openid`（`UserController.java`）
- token 通过 `token` 请求头传递

改完 Java 后要重新编译/重启，并让小程序重新登录，拿到带 openid 的新 token。

## 5. 启动 FastAPI 记忆服务

先确保 5000 端口没有被其他服务占用：

```powershell
netstat -ano | findstr :5000
```

然后：

```powershell
cd "D:\桌面\smart-housekeeping-ai-agent"
python -m uvicorn server.app:app --host 0.0.0.0 --port 5000
```

成功标志：

```
⏳ 正在连接 Postgres 并初始化 checkpointer...
✅ Redis 已连接，画像/会话列表缓存已启用
✅ LangGraph工作流图构建完成
✅ FastAPI 记忆服务启动完成
INFO:     Uvicorn running on http://0.0.0.0:5000
```

## 6. 小程序

1. 微信开发者工具按 `Ctrl+B` 重新编译
2. 确认 `app.js` 的 `aiBaseUrl` 仍是 `http://127.0.0.1:5000`
3. 客服页请求会自动带 `token` 请求头（已改 `customer-service.js`）
4. 如果之前登录过，建议清一次 `token` 并重新登录，确保 token 含 openid

## 7. 验证

### 健康检查

```powershell
curl.exe http://127.0.0.1:5000/api/health
```

期望：

```json
{"status":"healthy","postgres":true,"redis":true}
```

### 简单问题

> 日常保洁多少钱？

应仍由服务顾问回答。

### 复杂问题

> 我的订单现在是什么状态？我有哪些优惠券？重新预约两室一厅大概多少钱？

应看到步骤进度，最终回答覆盖三个子问题。

### 称呼偏好

> 以后叫我刘佳仪

然后问：

> 日常保洁多少钱？

回答应以“刘佳仪”开头。确认 Postgres：

```sql
SELECT user_id, preferred_name, updated_at FROM user_profile;
```

### 会话列表

```powershell
curl.exe -H "token: <你的token>" http://127.0.0.1:5000/api/sessions
```

### 重启不丢记忆

1. 完成一次对话，记下 session_id
2. `Ctrl+C` 停掉 FastAPI，再启动
3. 用同一个 session_id 继续问
4. 能记得之前的内容，说明 PostgresSaver 生效

## 8. 数据清理

```powershell
python -m server.cleanup --days 90
```

会删除超过 90 天的会话（checkpointer + thread_meta）。

## 9. 常见问题

| 现象 | 处理 |
|---|---|
| `psql` 不是命令 | 把 `E:\PostgreSQL\bin` 加入 PATH，或用全路径 |
| `Postgres 连接失败` | 检查 `POSTGRES_URI` 的用户/密码/库名，确认 5432 监听 |
| `unknown command 'HELLO'` | 已通过 `protocol=2` 解决；不要改回默认协议 |
| Redis 连接失败 | 确认 `redis-server.exe` 在跑，6379 监听 |
| 5000 被占用 | 停掉占用 5000 的进程，或换端口并同步修改小程序 `aiBaseUrl` |
| token 解析不到 openid | Java 重新编译/重启，小程序重新登录 |
| 回答里没有称呼 | 检查 `user_profile` 是否有记录，token 是否带 openid |
| 重启后记忆丢失 | 检查 Postgres 是否有 `checkpoints` 表，FastAPI 是否真的用了 PostgresSaver |

## 10. 注意

- FastAPI 占用 5000，不要同时运行其他服务占用该端口
- `thread_meta` 是给小程序客服用的会话列表；Java 的 `chat_session` 是管理端 AI 助手的会话表，两者独立
- `USER_JWT_SECRET` 与 Java `smart.jwt.user-secret-key` 必须一致
- Redis 只是缓存，删掉 Redis 数据不影响对话记忆；对话记忆在 Postgres
