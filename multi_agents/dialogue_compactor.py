"""
对话历史结构化压缩

当对话消息超过 DIALOGUE_KEEP_MESSAGES 条时，把更早的消息交给 LLM，
按 Pydantic Schema 输出结构化摘要；最近 N 条保持原文。

摘要写入 AgentState.dialogue_summary，已压缩的条数写入 summary_upto，
由 checkpointer 随 thread 持久化。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from config import (
    DIALOGUE_COMPRESS_ENABLED,
    DIALOGUE_KEEP_MESSAGES,
    HTTP_MAX_RETRIES,
    HTTP_TIMEOUT,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    OPENAI_THINKING,
)


class DialogueSummary(BaseModel):
    """对话压缩后的结构化摘要。"""

    summary: str = Field(
        description="用一两句话概括这段历史对话的主题和进展"
    )
    confirmed_facts: List[str] = Field(
        default_factory=list,
        description="已确认的事实，必须原样保留订单号、金额、券码、日期、家政员姓名",
    )
    user_needs: List[str] = Field(
        default_factory=list,
        description="用户提出但尚未完全解决的诉求",
    )
    commitments: List[str] = Field(
        default_factory=list,
        description="平台已经给出的处理方案、补偿或退款承诺",
    )
    open_questions: List[str] = Field(
        default_factory=list,
        description="还需要向用户确认的问题",
    )


SUMMARY_SYSTEM_PROMPT = """你是智能家政客服的历史对话压缩器。

你的任务：把【已有摘要】和【需要压缩的新对话】合并成一份结构化摘要，供后续客服继续对话使用。

必须遵守：
1. 订单号、金额、券码、日期、家政员姓名、补偿/退款承诺必须原样保留，不得改写或省略。
2. 只总结对话中明确出现的信息，禁止编造、推测或补充。
3. 用户已经解决的问题不要放进 user_needs；还没解决的才放。
4. 平台已经承诺的方案放进 commitments。
5. 如果已有摘要和新对话冲突，以新对话为准。
6. 摘要要简洁，不要复述寒暄。"""


def _build_chat_model():
    """构造用于压缩的 ChatOpenAI（低温度，独立于业务 LLM 客户端）。"""
    from langchain_openai import ChatOpenAI

    kwargs: Dict[str, Any] = {
        "model": OPENAI_MODEL,
        "api_key": OPENAI_API_KEY,
        "base_url": OPENAI_BASE_URL,
        "temperature": 0,
        "timeout": HTTP_TIMEOUT,
        "max_retries": min(2, max(1, HTTP_MAX_RETRIES)),
    }
    thinking = str(OPENAI_THINKING or "").lower()
    if thinking in ("disabled", "false", "off", "none", "0"):
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    elif thinking in ("enabled", "true", "on", "1"):
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
    return ChatOpenAI(**kwargs)


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
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


def _normalize_summary(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """把模型输出规范成可持久化的纯 JSON 结构。"""
    if not isinstance(data, dict):
        return None
    summary = str(data.get("summary") or "").strip()
    if not summary:
        return None

    def _list(key: str, limit: int = 20) -> List[str]:
        value = data.get(key)
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()][:limit]

    return {
        "summary": summary[:500],
        "confirmed_facts": _list("confirmed_facts"),
        "user_needs": _list("user_needs"),
        "commitments": _list("commitments"),
        "open_questions": _list("open_questions"),
    }


def _format_messages_for_summary(records: List[Dict[str, Any]]) -> str:
    lines = []
    for msg in records:
        role = "用户" if msg.get("is_user", True) else "AI"
        timestamp = msg.get("timestamp", "")
        content = str(msg.get("content", ""))
        lines.append(f"[{timestamp}] {role}: {content}")
    return "\n".join(lines)


def _invoke_summary(messages: List[Any]) -> Optional[Dict[str, Any]]:
    """优先用 Schema 模式（function calling），失败时回退 JSON prompt。"""
    model = _build_chat_model()

    # 1) Pydantic Schema 模式
    try:
        structured = model.with_structured_output(DialogueSummary, method="function_calling")
        result = structured.invoke(messages)
        if isinstance(result, DialogueSummary):
            return _normalize_summary(result.model_dump())
        if isinstance(result, dict):
            return _normalize_summary(result)
    except Exception as e:
        print(f"⚠️ Schema 压缩失败，尝试 JSON 回退: {e}")

    # 2) JSON prompt 回退
    try:
        schema_text = json.dumps(DialogueSummary.model_json_schema(), ensure_ascii=False)
        fallback_messages = list(messages) + [HumanMessage(content=(
            "请只输出符合以下 JSON Schema 的 JSON 对象，不要 Markdown，不要解释：\n"
            f"{schema_text}"
        ))]
        response = model.invoke(fallback_messages)
        raw = getattr(response, "content", "")
        data = _extract_json_object(raw or "")
        if data:
            return _normalize_summary(data)
    except Exception as e:
        print(f"⚠️ JSON 回退压缩失败: {e}")

    return None


def compress_dialogue_if_needed(state: Dict[str, Any]) -> Dict[str, Any]:
    """对话超过保留条数时，压缩更早的消息。

    Args:
        state: 当前图状态，需包含 persisted_dialogue / dialogue_summary / summary_upto。

    Returns:
        partial state：{"dialogue_summary": dict, "summary_upto": int}；
        不需要压缩或压缩失败时返回空 dict。
    """
    if not DIALOGUE_COMPRESS_ENABLED:
        return {}

    records = state.get("persisted_dialogue") or []
    if not isinstance(records, list) or not records:
        return {}

    keep = max(2, int(DIALOGUE_KEEP_MESSAGES))
    upto = max(0, int(state.get("summary_upto") or 0))
    pending = records[upto:]
    if len(pending) <= keep:
        return {}

    to_compress = pending[:-keep]
    if not to_compress:
        return {}

    existing = state.get("dialogue_summary") or {}
    existing_text = json.dumps(existing, ensure_ascii=False) if existing else "无"

    messages = [
        SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"【已有摘要】\n{existing_text}\n\n"
            f"【需要压缩的新对话】\n{_format_messages_for_summary(to_compress)}\n\n"
            f"【当前保留原文的最近消息数】{keep} 条\n"
            "请合并成一份新的结构化摘要。"
        )),
    ]

    summary = _invoke_summary(messages)
    if not summary:
        print("⚠️ 对话摘要压缩未产出结果，保留原历史")
        return {}

    print(
        f"🗜️ 对话压缩: {len(to_compress)} 条 → 结构化摘要，"
        f"summary_upto={upto + len(to_compress)}"
    )
    print("📝 压缩后内容：")
    print(format_dialogue_summary(summary))
    return {
        "dialogue_summary": summary,
        "summary_upto": upto + len(to_compress),
    }


def format_dialogue_summary(summary: Optional[Dict[str, Any]]) -> str:
    """把结构化摘要格式化成提示词文本。"""
    if not summary or not isinstance(summary, dict):
        return ""

    lines = ["【历史摘要】"]
    if summary.get("summary"):
        lines.append(f"概要：{summary['summary']}")

    for label, key in (
        ("已确认事实", "confirmed_facts"),
        ("用户诉求", "user_needs"),
        ("已承诺", "commitments"),
        ("待确认", "open_questions"),
    ):
        items = summary.get(key) or []
        if items:
            lines.append(f"{label}：" + "；".join(str(item) for item in items))

    return "\n".join(lines)
