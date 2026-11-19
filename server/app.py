"""
自建记忆服务（FastAPI + PostgresSaver + Redis 缓存）

启动：
    python -m uvicorn server.app:app --host 0.0.0.0 --port 5000

小程序仍然请求 /api/chat/stream、/api/chat，端口保持不变。
"""

from __future__ import annotations

import json
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from langgraph.checkpoint.postgres import PostgresSaver
from pydantic import BaseModel
from psycopg_pool import ConnectionPool

from multi_agent_customer_service import make_graph
from tools.query_tools import extract_preferred_name

from . import auth, cache, store
from .config import POSTGRES_URI


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ProfileUpdate(BaseModel):
    preferred_name: Optional[str] = None
    default_address: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⏳ 正在连接 Postgres 并初始化 checkpointer...")
    pool = ConnectionPool(
        conninfo=POSTGRES_URI,
        min_size=1,
        max_size=10,
        open=True,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    )
    try:
        pool.wait(timeout=15)
    except Exception as e:
        pool.close()
        raise RuntimeError(f"Postgres 连接失败，请检查 POSTGRES_URI: {e}")

    store.set_pool(pool)
    store.init_schema()

    saver = PostgresSaver(pool)
    saver.setup()

    graph = make_graph(checkpointer=saver)
    app.state.pool = pool
    app.state.saver = saver
    app.state.graph = graph
    print("✅ FastAPI 记忆服务启动完成")

    try:
        yield
    finally:
        pool.close()


app = FastAPI(title="智能家政记忆服务", lifespan=lifespan)

# ===== 同一会话串行执行 =====
# 前端允许在 answer_done 后立刻发下一条，但此时上一张图可能还在写 checkpoint。
# 同一个 thread_id 必须排队，否则两条图的读写会互相覆盖。
_session_locks: Dict[str, threading.Lock] = {}
_session_locks_guard = threading.Lock()


def _session_lock(tid: str) -> threading.Lock:
    with _session_locks_guard:
        lock = _session_locks.get(tid)
        if lock is None:
            lock = threading.Lock()
            _session_locks[tid] = lock
        return lock

CONSOLE_FILE = Path(__file__).with_name("console.html")


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """根路径直接跳到网页控制台。"""
    return RedirectResponse(url="/console")


@app.get("/console", response_class=HTMLResponse, include_in_schema=False)
def console() -> str:
    """本地网页控制台：流式输出 + 长期记忆 + 计划步骤。"""
    return CONSOLE_FILE.read_text(encoding="utf-8")


def _sse(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def current_user(
    token: Optional[str] = Header(default=None),
    authorization: Optional[str] = Header(default=None),
) -> auth.AuthUser:
    return auth.resolve_user(token, authorization)


def _apply_preference(message: str, user_id: Optional[str], profile: Dict[str, Any]) -> Dict[str, Any]:
    """从当前消息中提取称呼偏好，写入用户画像。"""
    if not user_id:
        return profile

    name = extract_preferred_name(message)
    if name:
        updated = dict(profile or {})
        updated["preferred_name"] = name
        store.save_profile(user_id, updated)
        return updated

    # "不要叫我韩湘子"（没有给出新名字）→ 清除称呼
    if "不要叫我" in message:
        updated = dict(profile or {})
        updated["preferred_name"] = None
        store.save_profile(user_id, updated)
        return updated

    return profile


def _build_input(req: ChatRequest, user: auth.AuthUser, tid: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": req.message}],
        "customer_query": req.message,
        "session_id": tid,
        "user_id": user.user_id or "",
        "user_profile": profile or {},
    }


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "postgres": store.ping(),
        "redis": cache.ping(),
    }


@app.post("/api/new_session")
def new_session() -> Dict[str, str]:
    return {"session_id": str(uuid.uuid4())}


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest, user: auth.AuthUser = Depends(current_user)):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message 不能为空")

    graph = app.state.graph
    tid = (req.session_id or "").strip() or str(uuid.uuid4())
    profile = store.get_profile(user.user_id)
    profile = _apply_preference(req.message, user.user_id, profile)
    store.upsert_thread_meta(tid, user.user_id, req.message)

    input_state = _build_input(req, user, tid, profile)
    config = {"configurable": {"thread_id": tid}}

    session_lock = _session_lock(tid)

    def generate():
        yield _sse({"session_id": tid, "thread_id": tid})

        got_token = False
        final_state: Dict[str, Any] = {}
        error: Optional[str] = None

        if not session_lock.acquire(timeout=180):
            yield _sse({"error": "上一条消息仍在处理，请稍后再试"})
            yield "data: [DONE]\n\n"
            return

        try:
            for mode, chunk in graph.stream(
                input_state,
                config=config,
                stream_mode=["custom", "values"],
            ):
                if mode == "custom" and isinstance(chunk, dict):
                    if chunk.get("type") == "token":
                        got_token = True
                    yield _sse(chunk)
                elif mode == "values" and isinstance(chunk, dict):
                    final_state = chunk
        except Exception as e:
            error = str(e)
            print(f"❌ 运行失败: {e}")
            yield _sse({"error": f"运行失败: {e}"})
        finally:
            session_lock.release()

        if error is None:
            done: Dict[str, Any] = {
                "done": True,
                "session_id": tid,
                "thread_id": tid,
                "agent": final_state.get("current_agent", ""),
                "query_type": final_state.get("query_type", ""),
                "complexity": final_state.get("complexity", "simple"),
                "execution_mode": final_state.get("execution_mode", ""),
            }
            if final_state.get("complexity") == "complex":
                done["plan_steps"] = [
                    {
                        "step_id": step.get("step_id"),
                        "title": step.get("title"),
                        "status": step.get("status"),
                    }
                    for step in (final_state.get("plan") or [])
                ]
            if not got_token:
                done["content"] = final_state.get("response", "")
            yield _sse(done)

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/chat")
def chat(req: ChatRequest, user: auth.AuthUser = Depends(current_user)):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message 不能为空")

    graph = app.state.graph
    tid = (req.session_id or "").strip() or str(uuid.uuid4())
    profile = store.get_profile(user.user_id)
    profile = _apply_preference(req.message, user.user_id, profile)
    store.upsert_thread_meta(tid, user.user_id, req.message)

    session_lock = _session_lock(tid)
    if not session_lock.acquire(timeout=180):
        raise HTTPException(status_code=409, detail="上一条消息仍在处理，请稍后再试")
    try:
        result = graph.invoke(
            _build_input(req, user, tid, profile),
            config={"configurable": {"thread_id": tid}},
        )
    finally:
        session_lock.release()
    return {
        "response": result.get("response", ""),
        "session_id": tid,
        "agent": result.get("current_agent", ""),
        "complexity": result.get("complexity", "simple"),
    }


@app.get("/api/sessions")
def list_sessions(user: auth.AuthUser = Depends(current_user)):
    sessions = store.list_threads(user.user_id)
    return {"sessions": sessions, "count": len(sessions)}


@app.get("/api/sessions/{thread_id}")
def get_session(thread_id: str, user: auth.AuthUser = Depends(current_user)):
    meta = store.get_thread_meta(thread_id)
    if not meta:
        raise HTTPException(status_code=404, detail="会话不存在")

    if meta.get("user_id") and meta["user_id"] != user.user_id:
        raise HTTPException(status_code=403, detail="无权访问该会话")

    snapshot = app.state.graph.get_state({"configurable": {"thread_id": thread_id}})
    values = snapshot.values or {}
    return {
        "session_id": thread_id,
        "title": meta.get("title"),
        "created_at": meta.get("created_at"),
        "updated_at": meta.get("updated_at"),
        "messages": values.get("persisted_dialogue", []),
        "dialogue_summary": values.get("dialogue_summary", {}),
        "plan": values.get("plan", []),
    }


@app.delete("/api/sessions/{thread_id}")
def delete_session(thread_id: str, user: auth.AuthUser = Depends(current_user)):
    meta = store.get_thread_meta(thread_id)
    if not meta:
        raise HTTPException(status_code=404, detail="会话不存在")

    if meta.get("user_id") and meta["user_id"] != user.user_id:
        raise HTTPException(status_code=403, detail="无权删除该会话")

    try:
        app.state.saver.delete_thread(thread_id)
    except Exception as e:
        print(f"⚠️ 删除 checkpointer thread 失败: {e}")

    store.delete_thread_meta(thread_id)
    return {"ok": True, "session_id": thread_id}


@app.get("/api/profile")
def get_profile(user: auth.AuthUser = Depends(current_user)):
    return {"profile": store.get_profile(user.user_id)}


@app.put("/api/profile")
def update_profile(payload: ProfileUpdate, user: auth.AuthUser = Depends(current_user)):
    if not user.user_id:
        raise HTTPException(status_code=401, detail="需要登录")

    profile = dict(store.get_profile(user.user_id) or {})
    if payload.preferred_name is not None:
        profile["preferred_name"] = payload.preferred_name or None
    if payload.default_address is not None:
        profile["default_address"] = payload.default_address or None
    if payload.preferences is not None:
        merged = dict(profile.get("preferences") or {})
        merged.update(payload.preferences)
        profile["preferences"] = merged

    store.save_profile(user.user_id, profile)
    return {"ok": True, "profile": profile}
