"""
智能体包初始化文件。

使用惰性导入：只有真正访问某个 Agent 时才加载对应模块。
这样智能搜索只导入 smart_search_agent 时，不会顺带加载
客服的 6 个专家、会话管理器等，保持两边解耦。
"""

import importlib
from typing import Any

_LAZY_IMPORTS = {
    "BaseAgent": ".base_agent",
    "ServiceAgent": ".service_agent",
    "HousekeeperAgent": ".housekeeper_agent",
    "OrderAgent": ".order_agent",
    "PaymentAgent": ".payment_agent",
    "ComplaintAgent": ".complaint_agent",
    "GeneralAgent": ".general_agent",
    "PlanExecuteAgent": ".complex_agent",
    "SmartSearchAgent": ".smart_search_agent",
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
