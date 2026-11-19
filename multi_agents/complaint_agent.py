"""
投诉与售后专家智能体
专门负责家政服务投诉、售后处理和人工转接
"""

from typing import Dict, List, Any
from langchain_core.messages import HumanMessage, SystemMessage
from .base_agent import BaseAgent


class ComplaintAgent(BaseAgent):
    agent_key = "complaint_agent"

    def __init__(self):
        super().__init__(
            name="投诉与售后专家",
            role="客户投诉、售后处理与人工转接",
            expertise=["服务投诉", "返工与补偿", "物品损坏", "投诉流程", "转人工"]
        )

        # 售后工单暂不支持在线创建：不落库、不调工具，只回答投诉与售后处理规则
        # TODO(MySQL): 后续接入订单数据（orders / order_detail）作为上下文
        # 静态处理规则统一走 RAG（knowledge/raw/complaint），此处字典仅作降级兜底
        self.complaint_database = {
            "服务问题": {
                "家政员迟到": "先联系家政员确认到达时间；超过约定时间可在订单详情申请改约或取消，平台按实际等待情况协调补偿",
                "家政员爽约": "立即联系人工客服，平台优先协调替补家政员；确认无法上门服务的，可取消订单并全额退款",
                "服务态度问题": "记录订单号、发生时间和具体情况，平台会回访核实并对家政员进行培训或处罚",
                "服务不达标": "拍照或录像留证，可在订单详情申请返工；无法返工的按实际情况协商部分退款",
                "物品损坏": "现场拍照留证，保留订单号和沟通记录，平台协助定损并按责任协商赔付",
                "临时加价": "拒绝线下私下转账，保留聊天或录音证据并联系人工客服，平台会核实处理"
            },
            "售后政策": {
                "响应时效": "工作时间人工客服 2 小时内响应，一般投诉 24 小时内给出处理方案",
                "返工": "服务完成后发现清洁不到位，可优先申请免费返工，具体时限以订单页规则为准",
                "退款": "未开始服务可申请全额退款；已开始服务按实际完成情况协商部分退款",
                "补偿方式": "根据核实结果提供返工、部分退款、优惠券补偿等方式，具体以平台处理方案为准",
                "责任认定": "平台会结合订单记录、现场照片、双方沟通记录综合判断"
            },
            "投诉流程": {
                "第一步": "保留证据：订单号、现场照片/视频、聊天记录",
                "第二步": "在订单详情提交售后/投诉，或直接联系人工客服",
                "第三步": "平台核实情况，必要时联系家政员和客户双方",
                "第四步": "给出处理方案（返工、退款、补偿等）并执行",
                "第五步": "处理完成后回访确认，客户可对结果进行评价"
            },
            "转人工": {
                "需要转人工的情况": "人身安全、财产损失、重复扣款、家政员失联、对处理结果不满意",
                "转人工方式": "在小程序『我的-关于我们』查看客服电话，或通过在线客服入口转接",
                "转接前准备": "准备好订单号和问题描述，能显著加快处理速度"
            }
        }

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """处理投诉与售后查询"""
        customer_query = state["customer_query"]
        session_id = state.get("session_id", "default")

        # 对话轮次由 classify / 外层节点写入 persisted_dialogue
        conversation_context = self._get_conversation_context(session_id, state)

        # 按路由结果决定：RAG、动态工具，或两者都做
        matched_info = self._build_agent_context(
            customer_query, state, self._match_complaint_info
        )

        # 构建系统提示并增强对话上下文说明
        base_system_prompt = f"""你是{self.name}，专门负责{self.role}。
        你的专业领域包括：{', '.join(self.expertise)}

        请以专业、克制、有同理心的态度处理客户投诉：
        1. 先回应客户情绪，表达理解和歉意，但不要未经核实就认定责任
        2. 帮客户梳理事实：订单号、服务时间、具体问题、是否留有证据
        3. 给出明确的处理路径和时间预期，不要空泛承诺
        4. 涉及人身安全、财产损失、重复扣款、家政员失联等情况，优先建议转人工客服
        5. 涉及迟到和爽约时必须分开说明：迟到 15-30 分钟补 20 元券，超过 30 分钟补 30 元券；
           爽约（未到场/失联）补 50 元券。不要混用金额。

        回答要真诚、具体。不要编造赔偿金额或处理结果，补偿方案以平台核实后的结论为准。"""

        system_prompt = self._enhance_system_prompt_with_context(base_system_prompt)

        # 构建消息列表
        messages = []

        # 添加对话历史上下文（如果有的话）
        if conversation_context:
            context_message = f"""对话历史上下文：
{conversation_context}

请基于以上对话历史和当前查询，提供连贯的处理方案。"""
            messages.append(SystemMessage(content=context_message))

        # 添加系统提示
        messages.append(SystemMessage(content=system_prompt))

        # 如果有匹配的投诉信息，添加到上下文中
        if matched_info:
            complaint_context = f"""投诉与售后处理规则：
{matched_info}

当前查询：{customer_query}"""
            messages.append(HumanMessage(content=complaint_context))
        else:
            messages.append(HumanMessage(content=customer_query))

        # 调用LLM（流式，token 会实时推送给前端）
        try:
            response_content = self._invoke_llm(messages)
        except Exception as e:
            print(f"投诉与售后专家调用LLM时出错: {e}")
            response_content = "抱歉，处理您的投诉时遇到技术问题，请稍后重试或联系人工客服。"

        state["response"] = response_content
        state["current_agent"] = self.name
        state["tools_used"].append(f"{self.name}_processing")

        return state

    def _match_complaint_info(self, query: str) -> str:
        """匹配查询中的投诉与售后信息"""
        query_lower = query.lower()
        matched_info: List[str] = []

        # 1. 按分类名命中
        for category, details in self.complaint_database.items():
            if category in query_lower:
                info_text = f"【{category}】\n"
                for key, value in details.items():
                    info_text += f"• {key}：{value}\n"
                matched_info.append(info_text)

        # 2. 按关键词命中对应分类
        keyword_map = {
            "服务问题": ["迟到", "爽约", "没来", "态度", "不干净", "不达标", "返工", "损坏", "弄坏",
                         "加价", "乱收费", "偷东西", "投诉"],
            "售后政策": ["赔偿", "补偿", "退款", "返工", "政策", "怎么赔", "怎么处理"],
            "投诉流程": ["流程", "怎么投诉", "在哪里投诉", "提交", "证据", "处理多久"],
            "转人工": ["人工", "客服电话", "转接", "没人管", "不满意", "失联"]
        }
        existing = [info.split("】")[0].replace("【", "") for info in matched_info]
        for category, keywords in keyword_map.items():
            if category in existing:
                continue
            if any(k in query_lower for k in keywords):
                info_text = f"【{category}】\n"
                for key, value in self.complaint_database[category].items():
                    info_text += f"• {key}：{value}\n"
                matched_info.append(info_text)

        # 3. 没有命中时给出投诉处理总览
        if not matched_info:
            info_text = "【投诉处理总览】\n"
            for category, details in self.complaint_database.items():
                info_text += f"• {category}：{', '.join(details.keys())}\n"
            matched_info.append(info_text)

        return "\n".join(matched_info)
