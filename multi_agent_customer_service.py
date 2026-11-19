"""
智能家政多智能体客服系统
使用LangGraph构建，包含服务顾问、家政员顾问、预约订单、支付优惠、投诉售后、平台客服等专家
基于OpenAI兼容API提供LLM能力
支持多轮对话和会话管理
"""

import os
import json
import requests
import time
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langgraph.graph import StateGraph
from langgraph.config import get_config

try:
    from langgraph.config import get_stream_writer
except Exception:  # pragma: no cover - 兼容旧版 langgraph
    get_stream_writer = None
from langchain_core.tools import tool
from pydantic import BaseModel

# 加载环境变量
load_dotenv()

# 导入配置
from config import *

# 导入智能体和工具
from multi_agents import (
    ServiceAgent, HousekeeperAgent, OrderAgent,
    PaymentAgent, ComplaintAgent, GeneralAgent, PlanExecuteAgent
)
from multi_agents.base_agent import sanitize_order_ids
from multi_agents.dialogue_compactor import compress_dialogue_if_needed
from tools import classify_query_detailed
from tools.tool_router import route_intent

# 导入会话管理器
from session_manager import LangChainSessionManager, default_session_manager

# 超出客服范围时的固定回复（护栏：不调用业务智能体）
OUT_OF_SCOPE_REPLY = (
    "抱歉，这里是智能家政客服，仅处理服务咨询、家政员、预约订单、支付优惠、"
    "投诉售后及平台使用类问题；请用一句话说明您的具体业务诉求，我很乐意协助。"
)

def _thinking_config() -> dict:
    """根据 OPENAI_THINKING 生成思考模式参数。

    disabled/false/off/none/0 -> 关闭思考
    enabled/true/on/1         -> 开启思考
    其他                      -> 不传参数，使用服务商默认行为
    """
    value = str(OPENAI_THINKING or "").strip().lower()
    if value in ("disabled", "false", "off", "none", "0"):
        return {"thinking": {"type": "disabled"}}
    if value in ("enabled", "true", "on", "1"):
        return {"thinking": {"type": "enabled"}}
    return {}


# 复杂路径的计划步骤
class PlanStep(TypedDict, total=False):
    step_id: int
    title: str
    description: str
    agent: str
    tool: str
    args: Dict[str, Any]
    status: str          # pending | running | done | error
    result: str
    source: str          # planner | replanner | fallback


# 定义状态类型
class AgentState(TypedDict):
    session_id: str
    messages: List[Any]
    current_agent: str
    customer_query: str
    query_type: str
    response: str
    tools_used: List[str]
    next_agent: str
    conversation_history: List[Any]
    memory: Optional[BaseChatMessageHistory]
    # 由图 checkpointer 持久化，跨 LangGraph 工作进程仍可续聊（内存 session_manager 无法做到）
    persisted_dialogue: List[Any]
    # 对话摘要压缩：dialogue_summary 存结构化摘要，summary_upto 表示前多少条已被压缩
    dialogue_summary: Dict[str, Any]
    summary_upto: int
    # 用户级长期记忆（由 FastAPI 从 Postgres 读取后注入）
    user_id: str
    user_profile: Dict[str, Any]
    # 简单路径路由结果：tool / rag / planner + need_rag
    route_decision: Dict[str, Any]
    # 本轮允许出现在回答里的订单号（用户当前输入 + 工具实时结果）
    live_order_ids: List[str]
    # ===== 复杂路径（Plan-and-Execute / ReAct）字段 =====
    intents: List[str]
    complexity: str          # simple | complex
    primary_intent: str
    intent_reason: str
    plan: List[PlanStep]
    step_index: int
    total_steps_executed: int
    replan_count: int
    plan_finished: bool
    replan_action: str       # continue | finish
    execution_mode: str      # native_react | deterministic | deterministic_fallback
    execution_trace: List[Dict[str, Any]]
    path_error: str
    final_answer: str

# OpenAI兼容API客户端类
class OpenAICompatibleClient:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = HTTP_TIMEOUT
        self.max_retries = HTTP_MAX_RETRIES
        self.headers = HTTP_HEADERS.copy()
        self.headers["Authorization"] = f"Bearer {api_key}"

        # 添加LangChain回调管理器所需的属性
        self.parent_run_id = None
        self.run_id = None
        self.tags = []
        self.metadata = {}
        self.handlers = []
        self.callback_manager = None
        self.inheritable_handlers = []
        self.inheritable_tags = []
        self.inheritable_metadata = {}

    def _format_messages(self, messages):
        """把 LangChain 消息对象转换为 OpenAI 兼容接口的 messages 数组。"""
        formatted_messages = []
        for msg in messages:
            if hasattr(msg, 'content'):
                # 处理LangChain消息对象
                if hasattr(msg, 'type'):
                    if msg.type == 'human':
                        formatted_messages.append({"role": "user", "content": msg.content})
                    elif msg.type == 'ai':
                        formatted_messages.append({"role": "assistant", "content": msg.content})
                    elif msg.type == 'system':
                        # 系统消息转换为用户消息
                        formatted_messages.append({"role": "user", "content": f"System instruction: {msg.content}"})
                    else:
                        formatted_messages.append({"role": "user", "content": msg.content})
                else:
                    # 默认作为用户消息处理
                    formatted_messages.append({"role": "user", "content": msg.content})
            else:
                # 处理字符串或其他类型
                formatted_messages.append({"role": "user", "content": str(msg)})
        return formatted_messages

    def invoke(self, messages, temperature=None):
        """调用OpenAI兼容API（非流式，保留给分类器等场景）。

        temperature 为空时使用服务商默认值；分类器传 0 保证路由稳定。
        """
        formatted_messages = self._format_messages(messages)

        # 构建请求payload
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            **_thinking_config(),
        }
        if temperature is not None:
            payload["temperature"] = temperature

        # 调试信息（不打印完整提示词，避免日志过长和泄露内部指令）
        print(f"🔍 LLM 请求: model={self.model}, messages={len(formatted_messages)}, thinking={OPENAI_THINKING}")

        # 重试机制
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=self.headers,
                    timeout=self.timeout
                )

                response.raise_for_status()
                result = response.json()

                # 提取响应内容
                if "choices" in result and len(result["choices"]) > 0:
                    message = result["choices"][0].get("message", {})
                    content = message.get("content", "")
                    return CustomResponse(content)
                else:
                    return CustomResponse("API response format error")

            except requests.exceptions.RequestException as e:
                print(f"❌ Debug: Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise Exception(f"API call failed: {e}")
                time.sleep(2 ** attempt)  # 指数退避

    def stream(self, messages):
        """调用OpenAI兼容API，逐 token 返回内容（真流式）。"""
        formatted_messages = self._format_messages(messages)

        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "stream": True,
            **_thinking_config(),
        }

        print(f"🔍 LLM 流式请求: model={self.model}, messages={len(formatted_messages)}, thinking={OPENAI_THINKING}")

        for attempt in range(self.max_retries):
            received_any = False
            try:
                with requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=self.headers,
                    timeout=(self.timeout, self.timeout),
                    stream=True,
                ) as response:
                    response.raise_for_status()

                    # 逐行解析上游 SSE：data: {"choices":[{"delta":{"content":"..."}}]}
                    for raw_line in response.iter_lines():
                        if not raw_line:
                            continue
                        line = raw_line.decode("utf-8", errors="ignore").strip()
                        if not line.startswith("data:"):
                            continue

                        data = line[len("data:"):].strip()
                        if data == "[DONE]":
                            return

                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        choices = chunk.get("choices") or []
                        if not choices:
                            continue

                        delta = choices[0].get("delta") or {}
                        content = delta.get("content")
                        if content:
                            received_any = True
                            yield content

                return

            except requests.exceptions.RequestException as e:
                print(f"❌ Debug: 流式请求第 {attempt + 1} 次失败: {e}")
                # 已经输出过内容就不再重试，避免前端收到重复文本
                if received_any or attempt == self.max_retries - 1:
                    raise Exception(f"Streaming API call failed: {e}")
                time.sleep(2 ** attempt)  # 指数退避

    def chat(self, messages):
        """兼容LangChain的chat方法"""
        return self.invoke(messages)

    # 添加LangChain回调管理器接口
    def bind(self, **kwargs):
        """绑定参数到客户端"""
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def with_config(self, config):
        """设置配置"""
        if hasattr(config, 'get'):
            for key, value in config.items():
                setattr(self, key, value)
        return self

class CustomResponse:
    def __init__(self, content):
        self.content = content

# 全局会话管理器（使用 LangChain 标准接口）
session_manager = default_session_manager

# 延迟初始化LLM
_llm_instance = None

def initialize_llm_client():
    """初始化OpenAI兼容API客户端"""
    if not OPENAI_API_KEY:
        raise ValueError("API密钥未设置")

    return OpenAICompatibleClient(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        model=OPENAI_MODEL
    )

def get_llm():
    """获取LLM实例，延迟初始化"""
    global _llm_instance
    if _llm_instance is None:
        try:
            if not OPENAI_API_KEY:
                print("❌ 错误: API密钥未设置，无法初始化LLM")
                _llm_instance = None
            else:
                _llm_instance = initialize_llm_client()
                print(f"✅ 成功初始化API客户端")
        except Exception as e:
            print(f"❌ 初始化API客户端失败: {e}")
            print("将使用模拟响应模式")
            _llm_instance = None
    return _llm_instance

# 初始化智能体
def initialize_agents():
    """初始化所有智能体"""
    agents = {
        "service_agent": ServiceAgent(),
        "housekeeper_agent": HousekeeperAgent(),
        "order_agent": OrderAgent(),
        "payment_agent": PaymentAgent(),
        "complaint_agent": ComplaintAgent(),
        "general_agent": GeneralAgent()
    }

    # 为每个智能体设置LLM和会话管理器
    for agent in agents.values():
        agent.set_llm(get_llm())  # 延迟获取LLM
        agent.set_session_manager(default_session_manager)

    return agents

# 定义查询分类节点
def classify_query_node(state: AgentState) -> AgentState:
    """Classify customer query"""
    try:
        cfg = get_config()
        tid = (cfg.get("configurable") or {}).get("thread_id")
        if tid:
            state["session_id"] = str(tid)
    except RuntimeError:
        pass

    # 初始化状态对象
    if "session_id" not in state or not state.get("session_id"):
        import uuid
        state["session_id"] = str(uuid.uuid4())

    if "tools_used" not in state:
        state["tools_used"] = []

    if "conversation_history" not in state:
        state["conversation_history"] = []

    if "persisted_dialogue" not in state or state.get("persisted_dialogue") is None:
        state["persisted_dialogue"] = []

    if "memory" not in state:
        state["memory"] = None

    if "next_agent" not in state:
        state["next_agent"] = ""

    if "messages" not in state:
        state["messages"] = []

    # 复杂路径默认值（兼容没有这些字段的旧线程）
    for key, default in (
        ("user_id", ""),
        ("user_profile", {}),
        ("dialogue_summary", {}),
        ("summary_upto", 0),
        ("intents", []),
        ("complexity", "simple"),
        ("primary_intent", "general_inquiry"),
        ("intent_reason", ""),
        ("plan", []),
        ("step_index", 0),
        ("total_steps_executed", 0),
        ("replan_count", 0),
        ("plan_finished", False),
        ("replan_action", ""),
        ("execution_mode", ""),
        ("execution_trace", []),
        ("path_error", ""),
        ("final_answer", ""),
        ("route_decision", {}),
        ("live_order_ids", []),
    ):
        if key not in state or state.get(key) is None:
            state[key] = default

    # 获取必需字段
    customer_query = state.get("customer_query", "")
    session_id = state["session_id"]

    if not customer_query:
        state["response"] = "Error: No customer query provided"
        state["query_type"] = "general_inquiry"
        return state

    # 多标签分类 + 复杂度判定（LLM 主判，关键词兜底）
    try:
        classification = classify_query_detailed(customer_query, get_llm())
    except Exception as e:
        print(f"Error in query classification: {e}")
        classification = {
            "intents": ["general_inquiry"],
            "complexity": "simple",
            "primary_intent": "general_inquiry",
            "reason": "分类异常兜底",
            "source": "fallback",
        }

    intents = list(classification.get("intents") or ["general_inquiry"])
    complexity = classification.get("complexity") or "simple"
    primary_intent = classification.get("primary_intent") or intents[0]

    # 混合 out_of_scope 与业务意图时，只保留业务意图
    if "out_of_scope" in intents and len(intents) > 1:
        intents = [i for i in intents if i != "out_of_scope"]
        primary_intent = intents[0]

    state["intents"] = intents
    state["complexity"] = complexity
    state["primary_intent"] = primary_intent
    state["intent_reason"] = classification.get("reason", "")
    state["query_type"] = primary_intent
    state["tools_used"].append("query_classification")
    print(
        f"🧭 分类: intents={intents}, complexity={complexity}, "
        f"source={classification.get('source', 'unknown')}"
    )

    # 写入由 checkpointer 持久化的对话（用户轮次）
    pd = list(state.get("persisted_dialogue") or [])
    pd.append({
        "content": str(customer_query),
        "is_user": True,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    state["persisted_dialogue"] = pd

    # 同步到内存 session_manager（仅同进程有效；可选）
    try:
        session_manager.add_message(session_id, str(customer_query), is_user=True)
    except Exception as e:
        print(f"Error adding user message to session: {e}")

    # 护栏：超出范围直接固定回复，不进入业务智能体
    if state["query_type"] == "out_of_scope":
        state["response"] = OUT_OF_SCOPE_REPLY
        state["current_agent"] = "智能客服"
        pd_oos = list(state.get("persisted_dialogue") or [])
        pd_oos.append({
            "content": OUT_OF_SCOPE_REPLY,
            "is_user": False,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        state["persisted_dialogue"] = pd_oos
        try:
            session_manager.add_message(session_id, OUT_OF_SCOPE_REPLY, is_user=False)
        except Exception as e:
            print(f"Error adding out_of_scope refusal to session: {e}")
        state["tools_used"].append("out_of_scope_refusal")
        state.update(compress_dialogue_if_needed(state))
        return state

    # 路由层：LLM 工具路由器（tool / rag / tool+rag / planner）
    agent_key = _SIMPLE_AGENT_BY_INTENT.get(primary_intent, "general_agent")
    if complexity == "complex":
        route_decision = {
            "action": "planner",
            "tool": None,
            "args": {},
            "need_rag": False,
            "source": "complexity",
        }
    else:
        try:
            route_decision = route_intent(
                agent_key, customer_query, state, get_llm()
            )
        except Exception as e:
            print(f"⚠️ 路由失败，回退 RAG: {e}")
            route_decision = {
                "action": "rag",
                "tool": None,
                "args": {},
                "need_rag": True,
                "source": "fallback",
            }

    state["route_decision"] = route_decision
    if route_decision.get("action") == "planner":
        state["complexity"] = "complex"

    print(
        f"🧭 路由: action={route_decision.get('action')}, "
        f"tool={route_decision.get('tool')}, "
        f"need_rag={route_decision.get('need_rag')}, "
        f"source={route_decision.get('source')}"
    )

    return state

# 定义智能体处理节点
def create_agent_node(agent_name: str):
    """创建智能体处理节点"""
    def agent_node(state: AgentState) -> AgentState:
        agents = initialize_agents()
        agent = agents.get(agent_name)
        if agent:
            # 获取会话上下文
            session_id = state["session_id"]
            state["conversation_history"] = list(state.get("persisted_dialogue") or [])

            # 处理查询
            result = agent.process(state)

            if not isinstance(result, dict):
                print(f"Agent {agent_name} returned non-dict result: {type(result)}")
                result = {"response": "Error: Agent processing failed", "current_agent": agent_name}

            if "response" not in result:
                print(f"Agent {agent_name} result missing 'response' field: {result}")
                result["response"] = "Error: No response from agent"

            # 代码兜底：回答里只保留本轮允许的订单号，防止历史旧订单串入
            result["response"] = sanitize_order_ids(
                str(result.get("response", "")),
                result.get("live_order_ids") or [],
            )

            # 助手轮次写入 checkpointer 状态
            pd = list(result.get("persisted_dialogue") or [])
            pd.append({
                "content": str(result["response"]),
                "is_user": False,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
            result["persisted_dialogue"] = pd

            # 同步到内存 session_manager（仅同进程有效；可选）
            try:
                session_manager.add_message(session_id, str(result["response"]), is_user=False)
            except Exception as e:
                print(f"Error adding AI message to session: {e}")

            # 对话超过保留条数时，压缩更早的消息（不阻塞回答，回答已流式输出完）
            result.update(compress_dialogue_if_needed(result))

            return result
        else:
            state["response"] = f"Error: Agent {agent_name} not found"
            return state
    return agent_node

# 定义最终响应节点
def final_response_node(state: AgentState) -> AgentState:
    """Generate final response"""
    current_agent = state["current_agent"]
    response = state["response"]

    # 回答已经生成完毕：先通知前端“可以收尾了”，
    # 不用等 checkpoint 写完、done 事件到达才恢复发送按钮。
    if get_stream_writer is not None:
        try:
            writer = get_stream_writer()
            if writer is not None:
                writer({
                    "type": "answer_done",
                    "agent": current_agent,
                    "content": response,
                })
        except Exception as e:
            print(f"⚠️ 推送回答完成事件失败: {e}")

    state["response"] = f"【{current_agent}】\n{response}"
    return state


# =============================================================================
# 复杂路径：Router + Plan-and-Execute + ReAct
# =============================================================================

_SIMPLE_AGENT_BY_INTENT = {
    "service_info": "service_agent",
    "housekeeper_info": "housekeeper_agent",
    "order_service": "order_agent",
    "payment_voucher": "payment_agent",
    "complaint": "complaint_agent",
    "general_inquiry": "general_agent",
}


def _run_complex_node(complex_agent: PlanExecuteAgent, method: str, state: AgentState) -> Dict[str, Any]:
    """统一调用复杂路径方法：注入 LLM、捕获异常，保证节点不抛出。"""
    try:
        complex_agent.set_llm(get_llm())
        complex_agent.set_session_manager(default_session_manager)
        return getattr(complex_agent, method)(state)
    except Exception as e:
        print(f"❌ 复杂路径 {method} 失败: {e}")
        if method == "plan":
            return {"plan": [], "path_error": f"plan_failed: {e}"}
        if method == "execute_step":
            plan = list(state.get("plan") or [])
            index = int(state.get("step_index") or 0)
            if index < len(plan):
                step = dict(plan[index])
                step["status"] = "error"
                step["result"] = f"执行异常: {e}"
                plan[index] = step
            return {
                "plan": plan,
                "step_index": index + 1,
                "total_steps_executed": int(state.get("total_steps_executed") or 0) + 1,
            }
        if method == "replan":
            return {"replan_action": "finish", "plan_finished": True}
        return {
            "response": "抱歉，处理您的问题时出现异常，请稍后重试或转人工客服。",
            "current_agent": "综合客服",
        }


def route_after_classify(state: AgentState) -> str:
    """simple → 原 6 专家；complex → planner；out_of_scope → 固定回复。"""
    query_type = state.get("query_type") or "general_inquiry"
    if query_type == "out_of_scope":
        return "final_response"

    simple_target = _SIMPLE_AGENT_BY_INTENT.get(query_type, "general_agent")
    if not COMPLEX_PATH_ENABLED:
        return simple_target

    decision = state.get("route_decision") or {}
    if (state.get("complexity") or "simple") == "complex" or decision.get("action") == "planner":
        return "planner"
    return simple_target


def route_after_planner(state: AgentState) -> str:
    if state.get("path_error"):
        return "complex_fallback"
    if not (state.get("plan") or []):
        return "complex_fallback"
    return "executor"


def route_after_executor(state: AgentState) -> str:
    if state.get("path_error"):
        return "complex_fallback"
    executed = int(state.get("total_steps_executed") or 0)
    if executed >= MAX_TOTAL_STEPS:
        return "aggregator"
    plan = state.get("plan") or []
    if int(state.get("step_index") or 0) < len(plan):
        return "executor"
    return "replanner"


def route_after_replanner(state: AgentState) -> str:
    if state.get("plan_finished") or (state.get("replan_action") or "") == "finish":
        return "aggregator"
    if int(state.get("replan_count") or 0) >= MAX_REPLANS:
        return "aggregator"
    if int(state.get("total_steps_executed") or 0) >= MAX_TOTAL_STEPS:
        return "aggregator"
    plan = state.get("plan") or []
    if int(state.get("step_index") or 0) < len(plan):
        return "executor"
    return "aggregator"


def create_complex_fallback_node(simple_nodes: Dict[str, Any]):
    """复杂路径系统性失败时，复用现有单专家节点作答。"""

    def fallback_node(state: AgentState) -> Dict[str, Any]:
        key = _SIMPLE_AGENT_BY_INTENT.get(state.get("primary_intent") or "", "general_agent")
        working = dict(state)
        working["fallback_used"] = True
        working["tools_used"] = list(state.get("tools_used") or []) + ["complex_fallback"]
        try:
            return simple_nodes[key](working)
        except Exception as e:
            print(f"⚠️ 复杂路径降级到 {key} 失败: {e}")
            try:
                return simple_nodes["general_agent"](working)
            except Exception as e2:
                print(f"❌ 降级到平台客服也失败: {e2}")
                return {
                    "response": "抱歉，客服系统暂时繁忙，请稍后重试或转人工客服。",
                    "current_agent": "智能客服",
                }

    return fallback_node

# 图表入口点
# 使用方式：在langgraph.json文件中增加以下配置，声明构建图的方式，硬编码方式实现。
# "graphs": {
#     "customer_service": "./multi_agent_customer_service.py:make_graph"
# },
# 也可以在langgraph.json文件中使用workflow配置化的方式定义图的结构，但功能相对简单，无法实现复杂的逻辑
def _normalize_checkpointer(checkpointer):
    """兼容 langgraph dev 传入的 checkpointer 配置字典。

    langgraph.json 里可以配置 checkpointer 的 TTL，langgraph dev 会把这段配置
    作为参数传给图的构建函数。它不是 BaseCheckpointSaver 实例，不能直接传给
    workflow.compile()。这里保留真正的 saver，其余配置交给运行时管理。
    """
    if checkpointer is None or checkpointer is True or checkpointer is False:
        return checkpointer

    try:
        from langgraph.checkpoint.base import BaseCheckpointSaver
        if isinstance(checkpointer, BaseCheckpointSaver):
            return checkpointer
    except Exception:
        # 旧版本 langgraph 没有 BaseCheckpointSaver 时，用鸭子类型判断
        if hasattr(checkpointer, "get_tuple") and hasattr(checkpointer, "put"):
            return checkpointer

    print(
        f"⚠️ 忽略非 checkpointer 配置: {type(checkpointer).__name__}，"
        "由 LangGraph 运行时管理持久化"
    )
    return None


def make_graph(checkpointer=None):
    """构建LangGraph工作流图。

    Args:
        checkpointer: 可选的自定义 checkpointer。自建 FastAPI 服务时传入 PostgresSaver；
                      `langgraph dev` 加载时不传，由运行时注入。
    """
    # 创建工作流图
    workflow = StateGraph(AgentState)

    # 复杂路径共享实例（缓存原生工具探测结果与 ReAct 子图）
    complex_agent = PlanExecuteAgent()

    # 简单路径：现有 6 个专家节点
    simple_nodes = {
        "service_agent": create_agent_node("service_agent"),
        "housekeeper_agent": create_agent_node("housekeeper_agent"),
        "order_agent": create_agent_node("order_agent"),
        "payment_agent": create_agent_node("payment_agent"),
        "complaint_agent": create_agent_node("complaint_agent"),
        "general_agent": create_agent_node("general_agent"),
    }

    # 添加节点
    workflow.add_node("classify_query", classify_query_node)
    for name, node in simple_nodes.items():
        workflow.add_node(name, node)
    workflow.add_node("planner", lambda s: _run_complex_node(complex_agent, "plan", s))
    workflow.add_node("executor", lambda s: _run_complex_node(complex_agent, "execute_step", s))
    workflow.add_node("replanner", lambda s: _run_complex_node(complex_agent, "replan", s))
    workflow.add_node("aggregator", lambda s: _run_complex_node(complex_agent, "aggregate", s))
    workflow.add_node("complex_fallback", create_complex_fallback_node(simple_nodes))
    workflow.add_node("final_response", final_response_node)

    # 设置入口点
    workflow.set_entry_point("classify_query")

    # 分类后：越界 → 固定回复；复杂 → 规划；简单 → 原专家
    workflow.add_conditional_edges(
        "classify_query",
        route_after_classify,
        {
            "service_agent": "service_agent",
            "housekeeper_agent": "housekeeper_agent",
            "order_agent": "order_agent",
            "payment_agent": "payment_agent",
            "complaint_agent": "complaint_agent",
            "general_agent": "general_agent",
            "planner": "planner",
            "final_response": "final_response",
        },
    )

    # 规划 → 执行（失败 → 单专家兜底）
    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        {"executor": "executor", "complex_fallback": "complex_fallback"},
    )

    # 执行：还有步骤继续执行，否则复盘
    workflow.add_conditional_edges(
        "executor",
        route_after_executor,
        {
            "executor": "executor",
            "replanner": "replanner",
            "aggregator": "aggregator",
            "complex_fallback": "complex_fallback",
        },
    )

    # 复盘：继续补充步骤或汇总
    workflow.add_conditional_edges(
        "replanner",
        route_after_replanner,
        {"executor": "executor", "aggregator": "aggregator"},
    )

    # 汇总 / 兜底 → 最终响应
    workflow.add_edge("aggregator", "final_response")
    workflow.add_edge("complex_fallback", "final_response")

    # 简单路径：所有专家都连接到最终响应
    for name in simple_nodes:
        workflow.add_edge(name, "final_response")

    # 设置结束点
    workflow.set_finish_point("final_response")

    # 编译工作流（自建服务传入 PostgresSaver；langgraph dev 传配置字典时忽略）
    app = workflow.compile(checkpointer=_normalize_checkpointer(checkpointer))

    print("✅ LangGraph工作流图构建完成")
    return app

# 创建默认工作流实例
if __name__ == "__main__":
    app = make_graph()
    print("🚀 多智能体客服系统启动成功！")
