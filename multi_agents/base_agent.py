"""
基础智能体类
所有专门智能体的基类
"""

import re
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from session_manager import LangChainSessionManager, default_session_manager

try:
    from langgraph.config import get_stream_writer
except Exception:  # pragma: no cover - 兼容旧版 langgraph
    get_stream_writer = None


def _format_user_profile(profile: Optional[Dict[str, Any]]) -> str:
    """把用户画像格式化成提示词文本；空画像返回空字符串。"""
    if not profile or not isinstance(profile, dict):
        return ""

    lines = ["【用户画像】"]
    if profile.get("preferred_name"):
        lines.append(f"称呼：{profile['preferred_name']}")
    if profile.get("default_address"):
        lines.append(f"默认地址：{profile['default_address']}")
    if profile.get("last_order_id"):
        lines.append(f"最近订单：{profile['last_order_id']}")

    preferences = profile.get("preferences")
    if isinstance(preferences, dict):
        for key, value in preferences.items():
            if value in (None, "", [], {}):
                continue
            lines.append(f"{key}：{value}")

    return "\n".join(lines) if len(lines) > 1 else ""


_ORDER_ID_RE = re.compile(r"(?:ORD|HS)\d{6,}", re.IGNORECASE)
_PHONE_RE = re.compile(r"400[-\s]?\d{3}[-\s]?\d{4}")


def sanitize_order_ids(text: str, allowed_ids: Optional[List[str]] = None) -> str:
    """只保留本轮用户提供或工具查到的订单号，其余替换为“该订单”。

    这是 Prompt 规则之外的代码兜底，防止模型从对话历史里翻出旧订单号。
    """
    allowed = {str(order_id).upper() for order_id in (allowed_ids or [])}

    def replace(match: re.Match) -> str:
        return match.group(0) if match.group(0).upper() in allowed else "该订单"

    return _ORDER_ID_RE.sub(replace, text or "")


def sanitize_history_context(text: str, current_query: str = "") -> str:
    """在把对话历史送给模型之前，清掉旧的订单号和旧客服电话。

    只保留当前问题里用户自己提到的订单号/电话；其他一律替换，
    避免模型在流式生成时直接把历史旧数据输出给用户。
    """
    current = current_query or ""
    allowed_ids = {item.upper() for item in _ORDER_ID_RE.findall(current)}

    def replace_order(match: re.Match) -> str:
        return match.group(0) if match.group(0).upper() in allowed_ids else "该订单"

    cleaned = _ORDER_ID_RE.sub(replace_order, text or "")
    if not _PHONE_RE.search(current):
        cleaned = _PHONE_RE.sub("", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned


DATA_TRUST_RULES = """

【数据可信度与历史引用规则】

1. 本轮【实时数据】是订单状态、金额、优惠券余额、家政员档期、服务价格的唯一可信来源。
   没有【实时数据】时，不得给出具体订单号、金额、余额、状态或档期。
2. 本轮【规则资料】只用于解释规则、政策、流程。
   用户问通用规则（退款规则、取消规则、补偿标准等）时，只回答规则本身，
   不要主动说“您那笔订单……”，除非用户当前明确问的是那一笔订单。
3. 对话历史只用于理解上下文和指代，不得当作当前业务数据。
   历史里出现过的订单号、金额、状态，除非本轮工具重新查到，
   否则不要复述、不要补全、不要推测、不要拿来举例。
4. 用户问“我的订单/这单/我的券”，但本轮没有查到实时数据时：
   如实说明“当前没有查到”，并引导用户查看订单详情或转人工；
   不得用历史里的旧订单代替。
5. 用户画像只用于称呼、默认地址等偏好，
   不得用于订单、金额、券状态等业务事实。
6. 当实时数据与历史信息冲突时，以本轮实时数据为准；
   实时数据为空时，以“未查到”为准，不以历史为准。
7. 当本轮没有实时订单数据时，不要引用历史中的服务时间、家政员姓名、金额等订单细节；
   只给通用操作指引。
8. 如果本轮没有可用的客服电话，不要留空白号码；
   统一说“请在小程序『我的-关于我们』查看客服电话”。
9. 会变化、按用户不同、涉及交易/状态/库存/承诺的信息，都不能从历史引用；
   历史只用于理解上下文和称呼。
   包括：订单、退款、优惠券、价格、家政员档期与资质、服务范围、支付、发票、
   地址、消息、工单、赔偿、营业与活动状态、库存名额、政策版本、客服承诺。
   以上信息一律以本轮【实时数据】或【规则资料】为准；
   没有就如实说“未查到”，不要用历史补全、推测或举例。"""


class BaseAgent(ABC):
    # 各业务智能体覆盖此属性，用于 RAG 知识库按业务域过滤
    agent_key = ""

    def __init__(self, name: str, role: str, expertise: List[str], session_manager: LangChainSessionManager = None):
        self.name = name
        self.role = role
        self.expertise = expertise
        self.llm = None  # 将在运行时注入
        # 默认复用全局会话管理器，避免每创建一个智能体就重复初始化
        self.session_manager = session_manager or default_session_manager

    def set_llm(self, llm):
        """设置LLM客户端"""
        self.llm = llm

    def set_session_manager(self, session_manager: LangChainSessionManager):
        """设置会话管理器"""
        self.session_manager = session_manager

    def _get_stream_writer(self):
        """获取 LangGraph 自定义流写入器；不在图节点上下文时返回 None。"""
        if get_stream_writer is None:
            return None
        try:
            return get_stream_writer()
        except Exception:
            return None

    def _invoke_llm(self, messages) -> str:
        """
        调用 LLM 并返回完整文本。
        优先使用流式接口，把每个 token 通过 LangGraph 的 custom 流推给前端；
        客户端不支持 stream 或流式失败时，自动回退到非流式 invoke。
        """
        if self.llm is None:
            raise RuntimeError("LLM 未初始化，请检查 .env 中的 API 配置")

        writer = self._get_stream_writer()
        stream_method = getattr(self.llm, "stream", None)

        if callable(stream_method):
            chunks = []
            try:
                for token in stream_method(messages):
                    if not token:
                        continue
                    chunks.append(token)
                    if writer is not None:
                        try:
                            writer({"type": "token", "content": token, "agent": self.name})
                        except Exception as e:
                            print(f"推送流式 token 失败: {e}")
                            writer = None
                if chunks:
                    text = "".join(chunks)
                    # 最后一个 token 已到手：立即通知前端回答结束，
                    # 不等专家节点返回、checkpoint 和 done 事件。
                    if writer is not None:
                        try:
                            writer({"type": "answer_done", "agent": self.name, "content": text})
                        except Exception as e:
                            print(f"推送回答完成事件失败: {e}")
                    return text
                print("⚠️ 流式接口未返回内容，回退到非流式调用")
            except Exception as e:
                # 已经输出过部分内容时直接返回，避免重试造成重复文本
                if chunks:
                    print(f"⚠️ 流式输出中断，返回已生成内容: {e}")
                    return "".join(chunks)
                print(f"⚠️ 流式调用失败，回退到非流式调用: {e}")

        response = self.llm.invoke(messages)
        text = getattr(response, "content", str(response))
        if writer is None:
            writer = self._get_stream_writer()
        if writer is not None:
            try:
                writer({"type": "answer_done", "agent": self.name, "content": text})
            except Exception as e:
                print(f"推送回答完成事件失败: {e}")
        return text

    @abstractmethod
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """处理客户查询的抽象方法"""
        pass

    def _get_conversation_context(
        self,
        session_id: str,
        state: Optional[Dict[str, Any]] = None,
        max_messages: int = 12,
    ) -> str:
        """只从图状态读取对话/摘要/画像；state 不可用时才回退 session_manager。

        这样对话记忆只有一个真源（checkpointer），避免第二份数据不一致，
        也避免回退路径使用"读取时刻"而不是消息实际时间。
        """
        if state is not None:
            from .dialogue_compactor import format_dialogue_summary

            records = list(state.get("persisted_dialogue") or [])
            summary = state.get("dialogue_summary") or {}
            upto = max(0, int(state.get("summary_upto") or 0))
            profile_text = _format_user_profile(state.get("user_profile"))

            context_lines = []
            if profile_text:
                context_lines.append(profile_text)

            formatted_summary = format_dialogue_summary(summary)
            if formatted_summary:
                context_lines.append(formatted_summary)

            for msg in records[upto:][-max_messages:]:
                role = "用户" if msg.get("is_user", True) else "AI"
                content = msg.get("content", "")
                timestamp = msg.get("timestamp", "")
                context_lines.append(f"[{timestamp}] {role}: {content}")

            return sanitize_history_context(
                "\n".join(context_lines),
                state.get("customer_query", ""),
            )

        # 只有 state 不可用（非图调用场景）才回退 session_manager
        try:
            conversation_context = self.session_manager.get_conversation_context(session_id, max_messages)

            if not conversation_context:
                return ""

            # 格式化对话历史
            context_lines = []
            for msg in conversation_context:
                role = "用户" if msg.get("is_user", True) else "AI"
                content = msg.get("content", "")
                timestamp = msg.get("timestamp", "")
                context_lines.append(f"[{timestamp}] {role}: {content}")

            return sanitize_history_context(
                "\n".join(context_lines),
                (state or {}).get("customer_query", ""),
            )
        except Exception as e:
            print(f"获取对话上下文时出错: {e}")
            return ""

    def _retrieve_knowledge(self, query: str, k: int = 4) -> str:
        """从 RAG 知识库检索本业务域的资料。

        检索失败或无结果时返回空字符串，调用方据此回退到原有的关键词匹配。

        Args:
            query: 用户查询。
            k: 返回条数。

        Returns:
            str: 带分类和出处的资料文本；无结果时为空字符串。
        """
        try:
            from knowledge import format_knowledge, search_knowledge

            pairs = search_knowledge(
                query,
                agent_key=self.agent_key or None,
                k=k,
            )
            return format_knowledge(pairs)
        except Exception as e:
            print(f"知识库检索失败，回退关键词匹配: {e}")
            return ""

    def _get_route_decision(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """读取路由结果；没有路由或误入 planner 时回退为只查 RAG。"""
        decision = state.get("route_decision") or {}
        if not decision:
            return {
                "action": "rag",
                "tool": None,
                "args": {},
                "need_rag": True,
                "source": "default",
            }
        if decision.get("action") == "planner":
            return {
                "action": "rag",
                "tool": None,
                "args": {},
                "need_rag": True,
                "source": "planner_fallback",
            }
        return decision

    def _run_route_tool(self, decision: Dict[str, Any]) -> str:
        """执行路由选中的动态工具，返回带【实时数据】标记的文本。"""
        try:
            from tools.business_tools import TOOLS_BY_AGENT, run_tool_safely

            tool_name = decision.get("tool")
            allowed = {
                tool.name
                for tool in TOOLS_BY_AGENT.get(self.agent_key, [])
                if tool.name != "search_knowledge_base"
            }
            if not tool_name or tool_name not in allowed:
                return ""

            result = run_tool_safely(tool_name, decision.get("args") or {})
            return f"【实时数据｜{tool_name}】\n{result}"
        except Exception as e:
            print(f"实时数据查询失败: {e}")
            return f"【实时数据查询失败】{e}"

    def _build_agent_context(
        self,
        query: str,
        state: Dict[str, Any],
        fallback_matcher=None,
    ) -> str:
        """按路由结果决定查 RAG、调工具，或两者都做。"""
        decision = self._get_route_decision(state)
        parts: List[str] = []
        tool_info = ""

        if decision.get("need_rag", True):
            rag_info = self._retrieve_knowledge(query)
            if not rag_info and fallback_matcher is not None:
                rag_info = fallback_matcher(query)
            if rag_info:
                parts.append(f"【规则资料】\n{rag_info}")

        if decision.get("action") == "tool":
            tool_info = self._run_route_tool(decision)
            if tool_info:
                parts.append(tool_info)

        # 记录本轮允许出现的订单号：用户当前输入 + 工具实时结果
        current_ids = set(_ORDER_ID_RE.findall(query or ""))
        tool_ids = set(_ORDER_ID_RE.findall(tool_info))
        state["live_order_ids"] = sorted(current_ids | tool_ids)

        return "\n\n".join(parts)

    def _add_message_to_session(self, session_id: str, message: str, is_user: bool = True):
        """添加消息到会话历史"""
        try:
            self.session_manager.add_message(session_id, message, is_user)
        except Exception as e:
            print(f"添加消息到会话时出错: {e}")

    def _append_persisted_turn(self, state: Dict[str, Any], content: str, is_user: bool) -> List[Dict[str, Any]]:
        """把一轮对话写入 persisted_dialogue，并同步内存会话管理器。

        返回新列表（copy-on-write），避免直接修改传入的 state 对象。
        """
        from datetime import datetime

        records = list(state.get("persisted_dialogue") or [])
        records.append({
            "content": str(content),
            "is_user": bool(is_user),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        try:
            self.session_manager.add_message(state.get("session_id", "default"), str(content), is_user)
        except Exception as e:
            print(f"同步会话管理器失败: {e}")
        return records

    def _enhance_system_prompt_with_context(self, base_prompt: str) -> str:
        """增强系统提示，添加对话上下文说明"""
        context_instruction = """

重要：请结合对话历史上下文，理解客户之前的问题和需求，提供连贯、个性化的回答。
如果这是多轮对话，请参考之前的对话内容，避免重复信息，并基于客户的新问题提供补充信息。
保持对话的连贯性和自然性，让客户感受到你理解他们的完整需求。

如果上下文中出现【实时数据】，具体订单状态、金额、券余额、档期、价格以【实时数据】为准。
如果出现【规则资料】，规则、政策、流程以【规则资料】为准。
实时数据查询失败时如实说明，不要编造。"""

        return base_prompt + context_instruction + DATA_TRUST_RULES

    def get_info(self) -> Dict[str, Any]:
        """获取智能体信息"""
        return {
            "name": self.name,
            "role": self.role,
            "expertise": self.expertise
        }
