"""
支付与优惠专家智能体
专门负责支付、退款到账、优惠券和金额计算问题
"""

from typing import Dict, List, Any
from langchain_core.messages import HumanMessage, SystemMessage
from .base_agent import BaseAgent


class PaymentAgent(BaseAgent):
    agent_key = "payment_agent"

    def __init__(self):
        super().__init__(
            name="支付与优惠专家",
            role="支付、退款与优惠券问题处理",
            expertise=["支付方式", "支付异常", "退款到账", "优惠券规则", "秒杀券", "金额计算"]
        )

        # TODO: 正式版应接入 voucher / user_voucher / 订单支付退款数据
        # 这里维护支付与优惠券的通用规则，具体金额和券状态以页面为准
        self.payment_database = {
            "支付方式": {
                "微信支付": "提交订单后在支付页选择微信支付，按提示完成付款",
                "支付宝支付": "如支付页提供支付宝入口，可选择支付宝完成付款",
                "支付结果": "支付成功后订单由『待付款』变为『待接单』，可在历史订单查看"
            },
            "支付异常": {
                "支付失败": "先检查网络、余额或银行卡状态；若已扣款但订单仍显示待付款，请勿重复支付，保留支付凭证并联系人工客服",
                "重复支付": "不要再次付款，保留两次支付凭证和订单号，联系人工客服核实退款",
                "支付超时": "待付款订单长时间未支付可能自动取消，具体以订单页提示为准",
                "无法唤起支付": "退出小程序重新进入，或检查微信支付权限；仍失败可尝试其他支付方式"
            },
            "退款": {
                "申请退款": "在订单详情点击申请退款或取消，选择退款原因后提交",
                "退款进度": "提交后订单进入『退款中』，可在订单详情查看审核状态",
                "到账时间": "审核通过后原路退回，微信/支付宝一般 1~3 个工作日到账，具体以支付渠道为准",
                "退款金额": "未开始服务通常可全额退款；已开始服务按实际服务情况协商部分退款",
                "退款去向": "退款原路返回原支付账户，不支持退到其他账户"
            },
            "优惠券": {
                "普通券": "在首页优惠券区点击“领取”，领取后可在『我的-我的优惠券』查看，下单结算时选择使用",
                "秒杀券": "限时限量领取，领取成功后可在『我的-我的优惠券』查看",
                "折扣规则": "折扣券按订单金额乘折扣率抵扣，例如 8 折券实际支付订单金额的 80%",
                "使用门槛": "每张券有适用服务、最低消费、有效期等限制，以券面规则为准",
                "能否叠加": "同一订单通常只能使用一张优惠券，不能多张叠加，是否可与其他活动同享以结算页为准",
                "查看我的券": "进入『我的-我的优惠券』，可查看未使用、已使用和已过期的券",
                "退款退券": "整单退款后优惠券是否退回，以券面规则和退款结果为准"
            },
            "金额计算": {
                "订单金额": "订单金额由所选服务项目、服务时长和家政员单价组成",
                "优惠抵扣": "结算页会自动计算优惠券抵扣后的实付金额",
                "价格疑问": "如对金额有疑问，请提供订单号，由人工客服核对明细"
            }
        }

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """处理支付和优惠券相关查询"""
        customer_query = state["customer_query"]
        session_id = state.get("session_id", "default")

        # 对话轮次由 classify / 外层节点写入 persisted_dialogue
        conversation_context = self._get_conversation_context(session_id, state)

        # 按路由结果决定：RAG、动态工具，或两者都做
        matched_info = self._build_agent_context(
            customer_query, state, self._match_payment_info
        )

        # 构建系统提示并增强对话上下文说明
        base_system_prompt = f"""你是{self.name}，专门负责{self.role}。
        你的专业领域包括：{', '.join(self.expertise)}

        请根据客户的问题提供准确的支付和优惠说明：
        - 支付问题先判断是「付不了」还是「付了没反应」，给出对应的排查步骤
        - 退款问题说明申请入口、当前状态和大致到账时间，不要承诺具体到账时刻
        - 优惠券问题说明领取方式、使用门槛、能否叠加，提醒以券面和结算页为准
        - 涉及具体订单的金额、扣款、退款进度时，如果上下文没有真实数据，
          请引导客户在订单详情查看或转人工核对，切勿编造金额和到账时间

        回答要简洁、准确，涉及资金安全时提醒客户不要重复支付、不要私下转账。"""

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

        # 如果有匹配的支付信息，添加到上下文中
        if matched_info:
            payment_context = f"""支付与优惠规则：
{matched_info}

当前查询：{customer_query}"""
            messages.append(HumanMessage(content=payment_context))
        else:
            messages.append(HumanMessage(content=customer_query))

        # 调用LLM（流式，token 会实时推送给前端）
        try:
            response_content = self._invoke_llm(messages)
        except Exception as e:
            print(f"支付与优惠专家调用LLM时出错: {e}")
            response_content = "抱歉，处理您的支付问题时遇到技术问题，请稍后重试。"

        state["response"] = response_content
        state["current_agent"] = self.name
        state["tools_used"].append(f"{self.name}_processing")

        return state

    def _match_payment_info(self, query: str) -> str:
        """匹配查询中的支付和优惠信息"""
        query_lower = query.lower()
        matched_info: List[str] = []

        # 1. 按分类名命中
        for category, details in self.payment_database.items():
            if category in query_lower:
                info_text = f"【{category}】\n"
                for key, value in details.items():
                    info_text += f"• {key}：{value}\n"
                matched_info.append(info_text)

        # 2. 按关键词命中对应分类
        keyword_map = {
            "支付方式": ["支付方式", "怎么付", "微信支付", "支付宝", "付款"],
            "支付异常": ["支付失败", "付不了", "扣款", "重复支付", "支付超时", "没反应", "打不开"],
            "退款": ["退款", "退钱", "到账", "原路", "退多少", "退款进度"],
            "优惠券": ["优惠券", "券", "折扣", "秒杀", "满减", "抵扣", "叠加", "领券", "过期"],
            "金额计算": ["金额", "多少钱", "价格", "费用", "结算", "总价", "明细"]
        }
        existing = [info.split("】")[0].replace("【", "") for info in matched_info]
        for category, keywords in keyword_map.items():
            if category in existing:
                continue
            if any(k in query_lower for k in keywords):
                info_text = f"【{category}】\n"
                for key, value in self.payment_database[category].items():
                    info_text += f"• {key}：{value}\n"
                matched_info.append(info_text)

        # 3. 没有命中时给出支付与优惠总览
        if not matched_info:
            info_text = "【支付与优惠总览】\n"
            for category, details in self.payment_database.items():
                info_text += f"• {category}：{', '.join(details.keys())}\n"
            matched_info.append(info_text)

        return "\n".join(matched_info)
