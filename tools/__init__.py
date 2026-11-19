"""
工具包初始化文件。

惰性导入：智能搜索只导入 tools.smart_search_tools 时，
不会顺带加载客服的 query_tools / business_tools。
"""

import importlib
from typing import Any

_LAZY_IMPORTS = {
    "classify_query": ".query_tools",
    "classify_query_detailed": ".query_tools",
    "keyword_classify": ".query_tools",
    "ALL_TOOLS": ".business_tools",
    "TOOL_REGISTRY": ".business_tools",
    "TOOLS_BY_AGENT": ".business_tools",
    "run_tool_safely": ".business_tools",
}

__all__ = list(_LAZY_IMPORTS)


def __getattr__(name: str) -> Any:
    module_path = _LAZY_IMPORTS.get(name)
    if not module_path:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_path, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
