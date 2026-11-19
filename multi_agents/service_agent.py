"""
服务顾问智能体
专门负责家政服务项目、套餐咨询与推荐
"""

import re
from typing import Dict, List, Any
from langchain_core.messages import HumanMessage, SystemMessage
from .base_agent import BaseAgent

_SERVICE_NOT_FOUND_RE = re.compile(r"没有找到(?:服务|套餐)[“\"]([^”\"]+)[”\"]")


class ServiceAgent(BaseAgent):
    agent_key = "service_agent"

    def __init__(self):
        super().__init__(
            name="服务顾问",
            role="家政服务项目与套餐咨询",
            expertise=["服务项目介绍", "套餐搭配", "价格说明", "服务流程", "适用场景推荐"]
        )

        # 服务项目、套餐、价格一律以工具查询 service_item / service_package 的实时结果为准，
        # 这里不再维护写死的服务清单，避免与业务库不一致。
        # 仅保留服务流程、计价方式等通用说明（不含具体服务名和价格）。
        self.service_notes = {
            "服务流程": "选择服务 → 选择家政员和预约时段 → 填写服务地址 → 提交订单 → 支付 → 等待接单 → 家政员上门服务 → 完成后评价",
            "价格说明": "页面价格为参考价，实际金额以预约页按房屋面积、服务时长和套餐组合计算后的结算价为准",
            "工具耗材": "基础保洁服务一般由家政员自备基础工具；如需指定清洁剂或特殊耗材，请在订单备注中说明",
            "服务范围": "具体可服务的城市、区域和时段以预约页实时展示为准，偏远区域可能暂不可预约",
            "改约取消": "下单后可在订单详情申请改约或取消，能否改约取决于目标时段是否可约，退款规则以订单页提示为准"
        }

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """处理服务咨询类查询"""
        customer_query = state["customer_query"]
        session_id = state.get("session_id", "default")

        # 对话轮次由 classify / 外层节点写入 persisted_dialogue
        conversation_context = self._get_conversation_context(session_id, state)

        # 按路由结果决定：RAG、动态工具，或两者都做
        matched_info = self._build_agent_context(
            customer_query, state, self._match_service_notes
        )
        # 工具明确返回“没有找到”时直接固定回复，避免 LLM 继续介绍或估算该服务
        not_found = _SERVICE_NOT_FOUND_RE.search(matched_info or "")
        if not_found:
            name = not_found.group(1)
            print(f"🛑 服务未找到，固定回复：{name}", flush=True)
            state["response"] = (
                f"当前服务目录中没有找到「{name}」这个服务或套餐。"
                "您可以在『家政预约』页查看现有项目和价格，"
                "或告诉我想要的服务类型，我帮您从现有服务里筛选；最终价格以下单页为准。"
            )
            state["current_agent"] = self.name
            state["tools_used"].append(f"{self.name}_not_found")
            return state

        # 构建系统提示并增强对话上下文说明
        base_system_prompt = f"""你是{self.name}，专门负责{self.role}。
        你的专业领域包括：{', '.join(self.expertise)}

        请根据客户的问题提供专业、清晰的服务介绍：
        - 说明服务包含哪些内容、适合什么房型或人群
        - 服务项目、套餐名称和价格必须来自【实时数据】的工具查询结果；没有实时数据时，不要列举具体项目或报价
        - 如果【实时数据】返回 ok=false 或“没有找到服务/套餐”，回答只能说明当前目录中没有该名称，并建议查看预约页或从已有服务中选择；不要介绍该服务的内容/适用场景，不要追问面积、时长，也不要估算价格
        - 涉及价格时给出参考价，并提醒最终以下单页结算价为准
        - 客户在多个服务之间犹豫时，主动对比差异并给出推荐
        - 客户需求不明确时，先追问房屋面积、服务频次、重点清洁区域等信息

        回答要口语化、简洁，不要堆砌无关的服务项目。如果客户询问的服务不在知识范围内，
        请引导客户在『家政预约』页查看最新服务，或转人工客服确认，切勿编造服务内容和价格。"""

        system_prompt = self._enhance_system_prompt_with_context(base_system_prompt)

        # 构建消息列表
        messages = []

        # 添加对话历史上下文（如果有的话）
        if conversation_context:
            context_message = f"""对话历史上下文：
{conversation_context}

请基于以上对话历史和当前查询，提供连贯的回答。"""
            messages.append(SystemMessage(content=context_message))

        # 添加系统提示
        messages.append(SystemMessage(content=system_prompt))

        # 如果有匹配的服务信息，添加到上下文中
        if matched_info:
            service_context = f"""服务项目资料：
{matched_info}

当前查询：{customer_query}"""
            messages.append(HumanMessage(content=service_context))
        else:
            messages.append(HumanMessage(content=customer_query))

        # 调用LLM（流式，token 会实时推送给前端）
        try:
            response_content = self._invoke_llm(messages)
        except Exception as e:
            print(f"服务顾问调用LLM时出错: {e}")
            response_content = "抱歉，处理您的服务咨询时遇到技术问题，请稍后重试。"

        state["response"] = response_content
        state["current_agent"] = self.name
        state["tools_used"].append(f"{self.name}_processing")

        return state

    def _match_service_notes(self, query: str) -> str:
        """兜底通用服务说明；服务项目、套餐和价格必须来自工具实时查询。"""
        query_lower = query.lower()
        keywords = ["流程", "怎么预约", "如何预约", "区别", "哪个好", "推荐",
                    "适合", "需要准备", "自备", "工具", "范围", "改约", "取消", "价格", "收费"]
        if not any(k in query_lower for k in keywords):
            return ""
        note_lines = [f"• {k}：{v}" for k, v in self.service_notes.items()]
        return "【服务说明】\n" + "\n".join(note_lines)
