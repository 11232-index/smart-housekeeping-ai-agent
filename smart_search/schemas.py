"""智能搜索 API 的请求/响应模型。"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SearchFilters(BaseModel):
    max_price: float = Field(default=0, ge=0, description="最高价格，0 表示不限")
    min_rating: float = Field(default=0, ge=0, le=5, description="最低评分，0 表示不限")
    date: str = Field(default="", description="指定日期 YYYY-MM-DD，留空表示明天起 3 天")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="用户自然语言需求")
    filters: Optional[SearchFilters] = None


class SearchResponse(BaseModel):
    ok: bool
    answer: str
    results: List[Dict[str, Any]] = []
    suggestions: List[str] = []
    decision: Dict[str, Any] = {}
