"""
复杂任务执行智能体（Plan-and-Execute + ReAct）

职责：
- plan:      把多意图问题拆成 1~N 个可执行步骤
- execute:   执行单步（优先原生 Function Calling，失败自动降级为确定性工具调用）
- replan:    复盘已完成步骤，决定继续/结束
- aggregate: 把各步骤结果汇总成最终客服回答（流式输出）

简单路径（single intent）仍由现有 6 个业务智能体处理，本类只服务复杂路径。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from config import (
    HTTP_MAX_RETRIES,
    HTTP_TIMEOUT,
    MAX_PLAN_STEPS,
    MAX_REPLANS,
    MAX_STEP_RESULT_CHARS,
    MAX_TOTAL_STEPS,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    OPENAI_THINKING,
    PLANNER_JSON_RETRIES,
    REACT_NATIVE_TOOLS,
    REACT_RECURSION_LIMIT,
)
from tools.business_tools import (
    ALL_TOOLS,
    INTENT_DEFAULT_TOOL,
    TOOL_CATALOG_FOR_PROMPT,
    TOOL_REGISTRY,
    TOOLS_BY_AGENT,
    infer_tool_for_step,
    run_tool_safely,
)
from .base_agent import BaseAgent, DATA_TRUST_RULES, sanitize_order_ids
from .dialogue_compactor import compress_dialogue_if_needed

AGENT_KEYS: Tuple[str, ...] = (
    "service_agent",
    "housekeeper_agent",
    "order_agent",
    "payment_agent",
    "complaint_agent",
    "general_agent",
)

INTENT_TO_AGENT: Dict[str, str] = {
    "service_info": "service_agent",
    "housekeeper_info": "housekeeper_agent",
    "order_service": "order_agent",
    "payment_voucher": "payment_agent",
    "complaint": "complaint_agent",
    "general_inquiry": "general_agent",
}

# 各专家角色的关键字，用于规划器没给 agent 时兜底
_AGENT_HINTS: Dict[str, Tuple[str, ...]] = {
    "service_agent": ("服务", "保洁", "清洗", "价格", "套餐", "维修", "月嫂", "育儿"),
    "housekeeper_agent": ("家政员", "阿姨", "保姆", "评分", "等级", "档期"),
    "order_agent": ("订单", "预约", "改约", "取消", "催单", "迟到", "退款申请"),
    "payment_agent": ("支付", "优惠券", "券", "退款到账", "扣款", "金额", "结算"),
    "complaint_agent": ("投诉", "售后", "补偿", "赔偿", "损坏", "返工", "态度"),
    "general_agent": ("登录", "账号", "地址", "通知", "定位", "小程序", "营业时间"),
}

_AGENT_INFO_CACHE: Dict[str, Dict[str, Any]] = {}

PLANNER_PROMPT = """你是智能家政客服的复杂任务规划器。把用户问题拆成 1~{max_steps} 个可执行步骤。

可用执行专家（agent 字段只能取以下值）：
- service_agent: 服务项目、套餐、价格、服务范围
- housekeeper_agent: 家政员资料、评分、等级、档期
- order_agent: 订单状态、预约、改约、取消、催单
- payment_agent: 支付、退款到账、优惠券、金额计算
- complaint_agent: 投诉、售后、补偿、返工、工单
- general_agent: 平台信息、账号、地址、通知、小程序使用

可用工具（需要事实数据时填写 tool 和 args；纯解释步骤可留空）：
{tool_catalog}

规则：
1. 每一步只做一件事，步骤之间不要重复。
2. 必须查询真实数据时使用工具；能用已有常识回答的步骤不要滥用工具。
3. agent 必须来自上面的列表；tool 必须是上面工具之一或空字符串。
4. 只输出 JSON，不要 Markdown，不要解释。
5. 静态知识（政策、规则、流程、补偿、退款规则、平台使用、FAQ、服务范围）
   必须使用 search_knowledge_base；
   具体订单、优惠券余额、家政员档期、服务价格等实时数据，才使用对应业务工具。
6. 用户没有提供订单号，也没有询问“我的订单/这单/订单进度/到哪一步/订单状态”时，
   不要调用 query_order；通用规则问题只查 search_knowledge_base。
   投诉、售后类问题只回答处理规则，不要承诺创建工单。
7. 对话历史只用于理解上下文，不得当作实时业务数据；
   不要因为历史提到订单就规划查询订单。

输出格式：
{{"goal":"一句话总目标","steps":[
  {{"step_id":1,"title":"步骤标题","description":"这一步要做什么","agent":"order_agent","tool":"query_order","args":{{}}}},
  {{"step_id":2,"title":"步骤标题","description":"","agent":"payment_agent","tool":"list_user_coupons","args":{{"user_id":"demo_user"}}}}
]}}"""

REPLANNER_PROMPT = """你是任务复盘器。根据原始问题和已执行步骤，判断信息是否足够回答用户。

输出严格 JSON（不要 Markdown、不要解释）：
{{"action":"continue 或 finish","reason":"一句话","new_steps":[]}}

规则：
- 信息已足够回答用户全部子问题 → finish。
- 仍缺少必要的子目标，且预算允许 → continue，并在 new_steps 里给出补充步骤（格式同 planner）。
- 不要重复已完成的步骤。
- 剩余的 replan 预算：{replan_left} 次；剩余可执行总步数：{steps_left} 步。
- 预算不足时直接 finish。"""

AGGREGATOR_PROMPT = """你是智能家政客服的最终回复者。请基于下面的计划执行结果回答用户。

规则：
1. 先结论后细节，按用户的每个子问题分段回答。
2. 只使用执行结果中的事实；订单号、金额、券码必须来自结果，不得编造。
3. 不要暴露工具名、JSON、内部步骤编号或“计划”等字眼。
4. 某一步查不到信息时，如实说明，并给出下一步建议（订单详情、转人工）。
5. 客服口吻，简洁口语化，不要客套堆砌。
6. 涉及迟到和爽约时必须分开表述：迟到 15-30 分钟补 20 元券，超过 30 分钟补 30 元券；
   爽约（未到场/失联）补 50 元券。不要把不同情形的金额混在一起。""" + DATA_TRUST_RULES

EXECUTOR_SYSTEM_PROMPT = """你是智能家政客服系统的任务执行器。你只负责完成当前这一个步骤。

- 需要事实数据时调用工具；工具报错就说明失败原因，不要编造。
- 只使用本轮工具结果；不要引用对话历史里的订单号、金额、状态作为事实。
- 不要给用户写最终客服答复，只返回可供汇总器使用的事实摘要。
- 不要重复无效调用；信息足够就结束。"""


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """从模型输出中提取第一个 JSON 对象。"""
    if not text:
        return None
    cleaned = str(text).strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _content_to_text(content: Any) -> str:
    """兼容字符串 / 内容块列表两种模型返回。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content or "")


def _agent_info(agent_key: str) -> Dict[str, Any]:
    """延迟实例化现有专家，取 name/role/expertise，避免重复写死。"""
    if agent_key in _AGENT_INFO_CACHE:
        return _AGENT_INFO_CACHE[agent_key]

    from .complaint_agent import ComplaintAgent
    from .general_agent import GeneralAgent
    from .housekeeper_agent import HousekeeperAgent
    from .order_agent import OrderAgent
    from .payment_agent import PaymentAgent
    from .service_agent import ServiceAgent

    mapping = {
        "service_agent": ServiceAgent,
        "housekeeper_agent": HousekeeperAgent,
        "order_agent": OrderAgent,
        "payment_agent": PaymentAgent,
        "complaint_agent": ComplaintAgent,
        "general_agent": GeneralAgent,
    }
    info = mapping.get(agent_key, GeneralAgent)().get_info()
    _AGENT_INFO_CACHE[agent_key] = info
    return info


def _guess_agent(text: str) -> str:
    """按关键字推断步骤该交给哪个专家。"""
    for agent_key, keywords in _AGENT_HINTS.items():
        if any(kw in text for kw in keywords):
            return agent_key
    return "general_agent"


_ORDER_ID_RE = re.compile(r"(?:ORD|HS)\d{6,}", re.IGNORECASE)
_ORDER_INTENT_KEYWORDS: Tuple[str, ...] = (
    "订单号", "我的订单", "这单", "订单进度", "到哪一步",
    "订单状态", "查订单", "查一下订单", "订单详情", "退款进度",
)


def _has_order_intent(text: str) -> bool:
    """判断用户是否在问某一笔具体订单，而不是通用规则。"""
    text = text or ""
    if _ORDER_ID_RE.search(text):
        return True
    return any(keyword in text for keyword in _ORDER_INTENT_KEYWORDS)


def _filter_unrequested_order_steps(
    steps: List[Dict[str, Any]],
    query: str,
) -> List[Dict[str, Any]]:
    """通用规则问题不查具体订单，避免带出无关订单数据。"""
    if _has_order_intent(query):
        return steps

    filtered = [
        step for step in steps
        if step.get("tool") != "query_order"
    ]
    for index, step in enumerate(filtered, 1):
        step["step_id"] = index
    return filtered


def _normalize_step(raw: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
    """校验并规范化单个计划步骤。"""
    if not isinstance(raw, dict):
        return None
    title = str(raw.get("title") or raw.get("description") or "").strip()
    if not title:
        return None

    agent = str(raw.get("agent") or "").strip()
    if agent not in AGENT_KEYS:
        agent = _guess_agent(title)

    tool_name = str(raw.get("tool") or "").strip()
    if tool_name and tool_name not in TOOL_REGISTRY:
        tool_name = ""

    raw_args = raw.get("args")
    args: Dict[str, Any] = {}
    if isinstance(raw_args, dict):
        args = {
            str(k): v for k, v in raw_args.items()
            if isinstance(v, (str, int, float, bool))
        }

    return {
        "step_id": index + 1,
        "title": title[:80],
        "description": str(raw.get("description") or "")[:300],
        "agent": agent,
        "tool": tool_name,
        "args": args,
        "status": "pending",
        "result": "",
        "source": "planner",
    }


def _validate_plan(data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """校验 planner 输出。"""
    raw_steps = data.get("steps") if isinstance(data, dict) else None
    if not isinstance(raw_steps, list) or not raw_steps:
        return None
    steps = [
        _normalize_step(raw, i)
        for i, raw in enumerate(raw_steps[:MAX_PLAN_STEPS])
    ]
    steps = [s for s in steps if s]
    if not steps:
        return None
    for i, step in enumerate(steps):
        step["step_id"] = i + 1
    return steps


def _validate_replan(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """校验 replanner 输出。"""
    if not isinstance(data, dict):
        return None
    action = str(data.get("action") or "").strip().lower()
    if action not in ("continue", "finish"):
        return None

    new_steps: List[Dict[str, Any]] = []
    if action == "continue":
        raw_steps = data.get("new_steps")
        if isinstance(raw_steps, list):
            new_steps = [
                _normalize_step(raw, i)
                for i, raw in enumerate(raw_steps[:MAX_PLAN_STEPS])
            ]
            new_steps = [s for s in new_steps if s]
        if not new_steps:
            action = "finish"

    return {
        "action": action,
        "reason": str(data.get("reason") or "")[:200],
        "new_steps": new_steps,
    }


class PlanExecuteAgent(BaseAgent):
    """复杂路径的规划、执行、复盘与汇总。"""

    def __init__(self):
        super().__init__(
            name="综合客服",
            role="复杂问题规划、工具执行与结果汇总",
            expertise=["多意图拆解", "跨业务查询", "工具编排", "结果汇总"],
        )
        self._react_agents: Dict[str, Any] = {}
        self._native_probe: Optional[Tuple[bool, str]] = None

    # ------------------------------------------------------------------
    # 抽象方法兜底（本类通过节点方法调用，不直接走 process）
    # ------------------------------------------------------------------
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state["response"] = "复杂路径请通过 plan/execute/replan/aggregate 节点调用"
        state["current_agent"] = self.name
        return state

    # ------------------------------------------------------------------
    # 规划
    # ------------------------------------------------------------------
    def plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = str(state.get("customer_query") or "").strip()
        context = self._get_conversation_context(state.get("session_id", "default"), state, max_messages=6)

        steps: List[Dict[str, Any]] = []
        if self.llm is not None and query:
            messages = [
                SystemMessage(content=PLANNER_PROMPT.format(
                    max_steps=MAX_PLAN_STEPS,
                    tool_catalog=TOOL_CATALOG_FOR_PROMPT,
                )),
                HumanMessage(content=(
                    f"用户问题：{query}\n"
                    f"识别意图：{state.get('intents') or []}\n"
                    f"对话上下文：\n{context or '无'}\n"
                    "请输出计划 JSON。"
                )),
            ]
            data = self._call_json(messages, _validate_plan, PLANNER_JSON_RETRIES)
            if data:
                steps = data

        steps = _filter_unrequested_order_steps(steps, query)

        if not steps:
            steps = self._fallback_plan(state)

        self._emit({
            "type": "plan",
            "steps": [
                {"step_id": s["step_id"], "title": s["title"], "status": "pending"}
                for s in steps
            ],
            "total": len(steps),
        })

        return {
            "plan": steps,
            "step_index": 0,
            "total_steps_executed": 0,
            "replan_count": 0,
            "plan_finished": not bool(steps),
            "replan_action": "",
            "execution_mode": "",
            "execution_trace": [],
            "path_error": "" if steps else "empty_plan",
        }

    def _fallback_plan(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """规划失败时退化为单步计划，保证复杂路径仍能回答。"""
        intent = str(state.get("primary_intent") or "general_inquiry")
        agent = INTENT_TO_AGENT.get(intent, "general_agent")
        tool_name = INTENT_DEFAULT_TOOL.get(intent, "")
        query = str(state.get("customer_query") or "")
        return [{
            "step_id": 1,
            "title": "处理用户问题",
            "description": query[:200],
            "agent": agent,
            "tool": tool_name,
            "args": {},
            "status": "pending",
            "result": "",
            "source": "fallback",
        }]

    # ------------------------------------------------------------------
    # 执行单步
    # ------------------------------------------------------------------
    def execute_step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        plan = list(state.get("plan") or [])
        index = int(state.get("step_index") or 0)
        executed = int(state.get("total_steps_executed") or 0)

        if not plan or index >= len(plan) or executed >= MAX_TOTAL_STEPS:
            return {"plan_finished": True, "replan_action": "finish"}

        step = dict(plan[index])
        step["status"] = "running"
        plan[index] = step
        self._emit({
            "type": "step",
            "stage": "start",
            "step_id": step["step_id"],
            "index": index,
            "title": step["title"],
            "status": "running",
            "agent": step["agent"],
            "tool": step.get("tool", ""),
            "total": len(plan),
        })

        mode = state.get("execution_mode") or self._initial_mode()
        try:
            result, mode = self._try_execute(state, step, mode)
            status = "done"
        except Exception as e:
            result = f"步骤执行失败: {e}"
            status = "error"
            mode = "deterministic_fallback" if mode == "native_react" else mode
            self._native_probe = (False, f"runtime_failed:{e}")

        step = dict(plan[index])
        step["status"] = status
        step["result"] = str(result)[:MAX_STEP_RESULT_CHARS]
        plan[index] = step

        self._emit({
            "type": "step",
            "stage": "end",
            "step_id": step["step_id"],
            "index": index,
            "title": step["title"],
            "status": status,
            "agent": step["agent"],
            "total": len(plan),
        })

        trace = list(state.get("execution_trace") or [])
        trace.append({
            "step_id": step["step_id"],
            "status": status,
            "agent": step["agent"],
            "tool": step.get("tool", ""),
            "mode": mode,
        })

        return {
            "plan": plan,
            "step_index": index + 1,
            "total_steps_executed": executed + 1,
            "execution_mode": mode,
            "execution_trace": trace[-50:],
        }

    def _try_execute(self, state: Dict[str, Any], step: Dict[str, Any], mode: str) -> Tuple[str, str]:
        """原生 ReAct 优先；失败立即用确定性方式重试本步，并锁定后续模式。"""
        if mode == "native_react":
            try:
                return self._run_react_step(state, step), mode
            except Exception as e:
                print(f"⚠️ 原生工具调用失败，改用确定性模式: {e}")
                self._native_probe = (False, f"runtime_failed:{e}")
                return self._run_deterministic_step(state, step), "deterministic_fallback"
        return self._run_deterministic_step(state, step), mode

    def _initial_mode(self) -> str:
        """根据 REACT_NATIVE_TOOLS 决定初始执行模式。"""
        value = (REACT_NATIVE_TOOLS or "auto").lower()
        if value == "off":
            return "deterministic"
        if value == "on":
            return "native_react"
        ok, _ = self._native_available()
        return "native_react" if ok else "deterministic"

    def _native_available(self) -> Tuple[bool, str]:
        """探测当前模型是否支持原生工具调用（进程内缓存一次）。"""
        if self._native_probe is not None:
            return self._native_probe
        try:
            from langchain_core.tools import tool as lc_tool

            @lc_tool
            def get_ping(city: str = "") -> str:
                """连通性探测工具。

                Args:
                    city: 城市名称，可留空。

                Returns:
                    固定字符串 pong。
                """
                return "pong"

            llm = self._build_chat_model()
            bound = llm.bind_tools([get_ping])
            response = bound.invoke([HumanMessage(content="请调用 get_ping 工具查询上海，不要直接回答。")])
            ok = bool(getattr(response, "tool_calls", None))
            self._native_probe = (ok, "ok" if ok else "no_tool_calls")
        except Exception as e:
            self._native_probe = (False, f"probe_error:{e}")
        print(f"🔧 原生工具调用探测: {self._native_probe}")
        return self._native_probe

    def _build_chat_model(self):
        """构造 ChatOpenAI（仅 ReAct 使用，避免影响简单路径）。"""
        from langchain_openai import ChatOpenAI

        kwargs: Dict[str, Any] = {
            "model": OPENAI_MODEL,
            "api_key": OPENAI_API_KEY,
            "base_url": OPENAI_BASE_URL,
            "temperature": 0.2,
            "timeout": HTTP_TIMEOUT,
            "max_retries": min(2, max(1, HTTP_MAX_RETRIES)),
        }
        thinking = str(OPENAI_THINKING or "").lower()
        if thinking in ("disabled", "false", "off", "none", "0"):
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        elif thinking in ("enabled", "true", "on", "1"):
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        return ChatOpenAI(**kwargs)

    def _build_react_agent(self, agent_key: str):
        """按专家角色构建 ReAct 子图（工具按角色限定）。"""
        if agent_key in self._react_agents:
            return self._react_agents[agent_key]

        from langgraph.prebuilt import create_react_agent

        info = _agent_info(agent_key)
        tools = TOOLS_BY_AGENT.get(agent_key) or ALL_TOOLS
        prompt = (
            f"{EXECUTOR_SYSTEM_PROMPT}\n\n"
            f"你的角色：{info['name']}，负责{info['role']}。\n"
            f"专业领域：{', '.join(info.get('expertise') or [])}。"
        )
        agent = create_react_agent(
            self._build_chat_model(),
            tools,
            prompt=prompt,
            version="v2",
        )
        self._react_agents[agent_key] = agent
        return agent

    def _run_react_step(self, state: Dict[str, Any], step: Dict[str, Any]) -> str:
        """用原生 Function Calling 执行一步。"""
        agent = self._build_react_agent(step.get("agent") or "general_agent")
        prior = self._format_past_steps(state)

        message = (
            f"用户原始问题：{state.get('customer_query', '')}\n"
            f"当前步骤：{step.get('title', '')}\n"
            f"步骤说明：{step.get('description', '') or '无'}\n"
            f"之前步骤的结果：\n{prior or '无'}\n"
            "请执行当前步骤，只返回事实摘要。"
        )

        result = agent.invoke(
            {"messages": [HumanMessage(content=message)]},
            config={"recursion_limit": REACT_RECURSION_LIMIT},
        )
        messages = (result or {}).get("messages") or []
        for msg in reversed(messages):
            if getattr(msg, "type", "") == "ai":
                text = _content_to_text(getattr(msg, "content", ""))
                if text.strip():
                    return text.strip()
        return "未能获取到该步骤的结果"

    def _run_deterministic_step(self, state: Dict[str, Any], step: Dict[str, Any]) -> str:
        """不依赖模型自主决策的确定性执行。"""
        tool_name = step.get("tool") or infer_tool_for_step(
            step.get("title", ""), state.get("customer_query", "")
        )
        if tool_name:
            args = step.get("args") or {}
            return f"[{tool_name}] {run_tool_safely(tool_name, args)}"

        # 纯解释步骤：用一次非流式 LLM 生成，不向前端推 token
        if self.llm is None:
            return "（未配置模型，无法生成该步骤内容）"
        info = _agent_info(step.get("agent") or "general_agent")
        messages = [
            SystemMessage(content=(
                f"你是{info['name']}，负责{info['role']}。"
                "请完成给定步骤，只返回事实摘要，不要客套。"
            )),
            HumanMessage(content=(
                f"用户问题：{state.get('customer_query', '')}\n"
                f"当前步骤：{step.get('title', '')}\n"
                f"步骤说明：{step.get('description', '') or '无'}\n"
                f"之前结果：\n{self._format_past_steps(state) or '无'}"
            )),
        ]
        response = self.llm.invoke(messages)
        return _content_to_text(getattr(response, "content", "")).strip() or "未获取到内容"

    def _format_past_steps(self, state: Dict[str, Any]) -> str:
        plan = state.get("plan") or []
        lines = []
        for step in plan:
            if step.get("status") not in ("done", "error"):
                continue
            lines.append(
                f"[{step.get('step_id')}] {step.get('title')} ({step.get('status')}): "
                f"{str(step.get('result') or '')[:800]}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 复盘
    # ------------------------------------------------------------------
    def replan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        executed = int(state.get("total_steps_executed") or 0)
        replans = int(state.get("replan_count") or 0)

        if executed >= MAX_TOTAL_STEPS or replans >= MAX_REPLANS:
            return {"replan_action": "finish", "plan_finished": True}

        if self.llm is None:
            return {"replan_action": "finish", "plan_finished": True}

        steps_left = max(0, MAX_TOTAL_STEPS - executed)
        replan_left = max(0, MAX_REPLANS - replans)

        messages = [
            SystemMessage(content=REPLANNER_PROMPT.format(
                replan_left=replan_left,
                steps_left=steps_left,
            )),
            HumanMessage(content=(
                f"用户问题：{state.get('customer_query', '')}\n"
                f"已执行步骤：\n{self._format_past_steps(state) or '无'}\n"
                "请输出复盘 JSON。"
            )),
        ]
        data = self._call_json(messages, _validate_replan, 1)
        if not data:
            return {"replan_action": "finish", "plan_finished": True}

        if data["action"] == "finish":
            return {
                "replan_action": "finish",
                "plan_finished": True,
            }

        new_steps = _filter_unrequested_order_steps(
            data["new_steps"], state.get("customer_query", "")
        )
        if not new_steps:
            return {"replan_action": "finish", "plan_finished": True}

        plan = list(state.get("plan") or [])
        next_id = max([int(s.get("step_id") or 0) for s in plan] or [0]) + 1
        for i, step in enumerate(new_steps):
            step["step_id"] = next_id + i
            step["status"] = "pending"
            step["result"] = ""
            step["source"] = "replanner"
            plan.append(step)

        return {
            "plan": plan,
            "replan_action": "continue",
            "plan_finished": False,
            "replan_count": replans + 1,
        }

    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------
    def aggregate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = str(state.get("customer_query") or "")
        past = self._format_past_steps(state)
        context = self._get_conversation_context(state.get("session_id", "default"), state, max_messages=6)

        answer = ""
        if self.llm is not None:
            messages = [
                SystemMessage(content=AGGREGATOR_PROMPT),
                HumanMessage(content=(
                    f"用户问题：{query}\n"
                    f"对话上下文：\n{context or '无'}\n"
                    f"执行结果：\n{past or '（没有可用的执行结果）'}\n"
                    "请给出最终客服回答。"
                )),
            ]
            try:
                answer = self._invoke_llm(messages)
            except Exception as e:
                print(f"❌ 汇总 LLM 调用失败: {e}")

        if not answer:
            answer = self._fallback_answer(state, past)

        # 代码兜底：只保留用户当前输入或本轮执行结果中的订单号
        allowed_ids = set(re.findall(r"(?:ORD|HS)\d{6,}", query, re.IGNORECASE))
        allowed_ids.update(re.findall(r"(?:ORD|HS)\d{6,}", past, re.IGNORECASE))
        answer = sanitize_order_ids(answer, sorted(allowed_ids))

        persisted = self._append_persisted_turn(state, answer, is_user=False)
        result = {
            "response": answer,
            "final_answer": answer,
            "current_agent": self.name,
            "persisted_dialogue": persisted,
        }
        # 对话超过保留条数时，压缩更早的消息
        merged_state = dict(state)
        merged_state["persisted_dialogue"] = persisted
        result.update(compress_dialogue_if_needed(merged_state))
        return result

    def _fallback_answer(self, state: Dict[str, Any], past: str) -> str:
        """模型不可用时，用执行结果拼一个确定性回答。"""
        if not past:
            return "抱歉，这个问题暂时没能查到足够的信息，建议您在订单详情查看，或转人工客服处理。"
        return "根据查询结果：\n" + past

    # ------------------------------------------------------------------
    # 通用
    # ------------------------------------------------------------------
    def _call_json(self, messages, validator, retries: int) -> Optional[Any]:
        """调用模型并解析 JSON，失败追加修正提示重试。"""
        for attempt in range(max(0, retries) + 1):
            try:
                response = self.llm.invoke(messages)
                raw = _content_to_text(getattr(response, "content", ""))
                data = _extract_json_object(raw)
                if data is not None:
                    validated = validator(data)
                    if validated is not None:
                        return validated
            except Exception as e:
                print(f"结构化输出调用失败（第 {attempt + 1} 次）: {e}")

            if attempt < retries:
                messages = list(messages) + [HumanMessage(
                    content="上次输出不是合法 JSON。请只输出 JSON 对象，不要 Markdown，不要解释。"
                )]
        return None

    def _emit(self, payload: Dict[str, Any]) -> None:
        """向前端推送计划/步骤进度事件。"""
        writer = self._get_stream_writer()
        if writer is None:
            return
        try:
            writer(payload)
        except Exception as e:
            print(f"推送进度事件失败: {e}")
