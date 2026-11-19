"""
预约与订单专家智能体
专门负责预约流程、订单状态、取消改约与催单问题
"""

from typing import Dict, List, Any
from langchain_core.messages import HumanMessage, SystemMessage
from .base_agent import BaseAgent


class OrderAgent(BaseAgent):
    agent_key = "order_agent"

    def __init__(self):
        super().__init__(
            name="预约与订单专家",
            role="预约流程与订单状态处理",
            expertise=["预约流程", "订单状态", "改约取消", "催单", "退款申请", "服务当天问题"]
        )

        # TODO: 正式版应接入订单查询工具（/user/order/page、/user/order/{id}）
        # 这里先维护流程和状态说明，具体订单数据以订单页为准
        self.order_database = {
            "预约流程": {
                "选择服务": "在『家政预约』页选择服务项目或套餐，加入预约清单",
                "选择家政员": "按评分、价格、等级筛选家政员，并选择可预约的日期和时段",
                "填写信息": "选择服务地址、联系人、手机号，可填写服务要求或备注",
                "提交订单": "在预约清单确认明细和金额后提交，订单状态变为『待付款』",
                "完成支付": "支付成功后进入『待接单』，等待家政员或平台接单",
                "上门服务": "家政员接单后按预约时间上门，服务开始后订单变为『服务中』"
            },
            "订单状态": {
                "1 待付款": "订单已提交但还未支付，请在订单详情完成支付，超时可能自动取消",
                "2 待接单": "已支付，正在等待家政员或平台接单",
                "3 已接单": "家政员已接单，将按预约时间上门服务",
                "4 服务中": "家政员已上门，服务正在进行中",
                "5 已完成": "服务已完成，可查看订单详情并评价家政员",
                "6 退款中": "退款申请已提交，正在审核或原路退回中",
                "7 已取消": "订单已取消，如已支付请关注退款进度"
            },
            "改约与取消": {
                "申请入口": "进入『我的-历史订单』，打开对应订单详情，选择改约或取消",
                "能否改约": "取决于目标日期和时段是否可约，以及家政员档期，提交后以审核结果为准",
                "取消规则": "未开始服务前一般可申请取消；已开始服务或临近上门时间，可能产生部分费用",
                "取消原因": "取消时需选择原因，便于平台统计和改进服务",
                "退款": "取消后如涉及退款，订单进入『退款中』，到账时间以支付渠道为准"
            },
            "催单与服务当天": {
                "催单": "在订单详情点击『催单』，平台会提醒家政员尽快响应",
                "家政员迟到": "先电话联系家政员确认；超过约定时间可在订单详情申请改约或取消",
                "联系不上": "可通过订单详情查看联系方式，仍无法联系请转人工客服",
                "修改地址": "服务开始前可在订单详情或联系客服修改，是否支持以页面提示为准",
                "临时加项": "服务当天新增项目需与家政员协商，并在平台补下单，避免线下私下交易"
            }
        }

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """处理预约和订单相关查询"""
        customer_query = state["customer_query"]
        session_id = state.get("session_id", "default")

        # 对话轮次由 classify / 外层节点写入 persisted_dialogue
        conversation_context = self._get_conversation_context(session_id, state)

        # 按路由结果决定：RAG、动态工具，或两者都做
        matched_info = self._build_agent_context(
            customer_query, state, self._match_order_info
        )

        # 构建系统提示并增强对话上下文说明
        base_system_prompt = f"""你是{self.name}，专门负责{self.role}。
        你的专业领域包括：{', '.join(self.expertise)}

        请根据客户的问题给出清晰的流程和下一步操作指引：
        - 先判断客户是在问「怎么预约」还是「已有订单出了什么问题」
        - 问流程时按步骤说明；问订单时先确认订单号和当前状态
        - 涉及具体订单的实时状态、金额、退款进度时，如果上下文没有真实订单数据，
          请引导客户在『我的-历史订单』查看，切勿编造订单状态或到账时间
        - 需要平台介入的情况，主动建议客户转人工客服并保留订单号

        回答要具体、可执行，少说套话。"""

        system_prompt = self._enhance_system_prompt_with_context(base_system_prompt)

        # 构建消息列表
        messages = []

        # 添加对话历史上下文（如果有的话）
        if conversation_context:
            context_message = f"""对话历史上下文：
{conversation_context}

请基于以上对话历史和当前查询，提供连贯的解答。"""
            messages.append(SystemMessage(content=context_message))

        # 添加系统提示
        messages.append(SystemMessage(content=system_prompt))

        # 如果有匹配的订单信息，添加到上下文中
        if matched_info:
            order_context = f"""预约与订单规则：
{matched_info}

当前查询：{customer_query}"""
            messages.append(HumanMessage(content=order_context))
        else:
            messages.append(HumanMessage(content=customer_query))

        # 调用LLM（流式，token 会实时推送给前端）
        try:
            response_content = self._invoke_llm(messages)
        except Exception as e:
            print(f"预约与订单专家调用LLM时出错: {e}")
            response_content = "抱歉，处理您的订单问题时遇到技术问题，请稍后重试。"

        state["response"] = response_content
        state["current_agent"] = self.name
        state["tools_used"].append(f"{self.name}_processing")

        return state

    def _match_order_info(self, query: str) -> str:
        """匹配查询中的预约和订单信息"""
        query_lower = query.lower()
        matched_info: List[str] = []

        # 1. 按分类名命中
        for category, details in self.order_database.items():
            if category in query_lower:
                info_text = f"【{category}】\n"
                for key, value in details.items():
                    info_text += f"• {key}：{value}\n"
                matched_info.append(info_text)

        # 2. 按关键词命中对应分类
        keyword_map = {
            "预约流程": ["怎么预约", "如何预约", "预约流程", "下单", "提交订单", "预约清单", "怎么用"],
            "订单状态": ["待付款", "待接单", "已接单", "服务中", "已完成", "退款中", "已取消",
                         "状态", "到哪一步", "进度", "什么意思"],
            "改约与取消": ["改约", "改时间", "改期", "取消", "退单", "不想做了", "退款申请"],
            "催单与服务当天": ["催单", "迟到", "没来", "联系不上", "地址", "改地址", "加项", "临时"]
        }
        existing = [info.split("】")[0].replace("【", "") for info in matched_info]
        for category, keywords in keyword_map.items():
            if category in existing:
                continue
            if any(k in query_lower for k in keywords):
                info_text = f"【{category}】\n"
                for key, value in self.order_database[category].items():
                    info_text += f"• {key}：{value}\n"
                matched_info.append(info_text)

        # 3. 没有命中时给出预约流程总览
        if not matched_info:
            info_text = "【预约流程】\n"
            for key, value in self.order_database["预约流程"].items():
                info_text += f"• {key}：{value}\n"
            matched_info.append(info_text)

        return "\n".join(matched_info)
