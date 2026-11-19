"""智能搜索独立服务（端口 5001）。

与智能客服的 server/app.py 完全分开：
- 不加载客服 Agent / 路由 / 会话
- 只调用 SmartSearchAgent + Qdrant + MySQL 只读

启动：
    python -m uvicorn smart_search.app:app --host 0.0.0.0 --port 5001
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse

from multi_agents.smart_search_agent import SmartSearchAgent
from smart_search.config import HOUSEKEEPERS_COLLECTION, SERVICES_COLLECTION
from smart_search.qdrant_store import ensure_all_collections, get_client
from smart_search.schemas import SearchRequest, SearchResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 只保证集合存在，不自动重建索引（重建用 indexer 脚本）
    ensure_all_collections(recreate=False)
    app.state.agent = SmartSearchAgent()
    yield


app = FastAPI(title="智能家政智能搜索", version="1.0.0", lifespan=lifespan)
CONSOLE_FILE = Path(__file__).with_name("console.html")

# 开发阶段允许小程序开发者工具跨域；生产建议收紧
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/console")


@app.get("/console", response_class=HTMLResponse, include_in_schema=False)
def console() -> str:
    return CONSOLE_FILE.read_text(encoding="utf-8")


@app.get("/api/smart-search/health")
def health() -> Dict[str, Any]:
    try:
        client = get_client()
        services = client.get_collection(SERVICES_COLLECTION)
        housekeepers = client.get_collection(HOUSEKEEPERS_COLLECTION)
        return {
            "status": "healthy",
            "qdrant": True,
            "services_points": services.points_count,
            "housekeepers_points": housekeepers.points_count,
        }
    except Exception as exc:
        return {"status": "degraded", "qdrant": False, "error": str(exc)}


@app.post("/api/smart-search/search", response_model=SearchResponse)
def search(req: SearchRequest) -> Dict[str, Any]:
    filters = req.filters.model_dump() if req.filters else None
    return app.state.agent.process(req.query, filters)
