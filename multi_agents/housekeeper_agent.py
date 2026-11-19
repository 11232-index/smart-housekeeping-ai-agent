"""
家政员顾问智能体
专门负责家政员资料、等级、评价与预约时段咨询
"""

from typing import Dict, List, Any
from langchain_core.messages import HumanMessage, SystemMessage
from .base_agent import BaseAgent


class HousekeeperAgent(BaseAgent):
    agent_key = "housekeeper_agent"

    def __init__(self):
        super().__init__(
            name="家政员顾问",
            role="家政员资料咨询与匹配推荐",
            expertise=["家政员等级", "评分与评价", "擅长领域", "资质经验", "可预约时段"]
        )

        # TODO: 正式版应从 housekeeper / employee_detail / housekeeper_schedule 表读取
        # 这里整理的是平台通用的筛选与等级规则，具体家政员资料以详情页为准
        self.housekeeper_database = {
            "家政员等级": {
                "金牌家政员": "综合评分高、服务经验丰富，适合对服务品质要求较高的家庭，单价相对更高",
                "银牌家政员": "经验较丰富、评分稳定，性价比较好，适合大多数家庭日常服务",
                "铜牌家政员": "具备基础服务经验，适合日常简单保洁等标准化服务",
                "无牌家政员": "新入驻或基础家政员，价格相对较低，可按评价和详情页信息选择"
            },
            "筛选维度": {
                "评分": "家政员详情页展示综合评分，评分越高代表历史用户评价越好",
                "价格": "每位家政员有独立单价，可在预约页按价格高低筛选",
                "擅长领域": "可查看家政员擅长的服务类型，如日常保洁、家电清洗、月嫂育儿、老人陪护",
                "经验与资质": "详情页展示从业经验、健康证、培训或技能证书等信息",
                "用户评价": "可查看历史服务评价，了解服务态度和实际口碑"
            },
            "可预约时段": {
                "可预约范围": "只展示明天起 3 个自然日的时段（明天、后天、大后天），不支持当天预约",
                "查看档期": "进入家政员详情页或预约页，可查看明天起 3 天内可预约的日期和时段（如 09:00~11:00）",
                "时段已满": "显示不可约表示该时段已被预约，可选择其他时段或其他家政员",
                "改约": "下单后如需改约，可在订单详情发起申请，能否改约取决于目标时段是否仍可约",
                "提前预约": "热门家政员和周末时段较紧张，建议提前 1~3 天预约"
            },
            "选择建议": {
                "看需求": "日常保洁优先看评分和准时率；月嫂、育儿嫂、老人陪护优先看资质、经验和健康证",
                "看预算": "预算有限可选铜牌/银牌家政员，对服务品质要求高可选金牌家政员",
                "看评价": "重点查看与自家需求相似的历史评价，而不只看星级",
                "看档期": "先确认家政员在目标日期是否可约，再提交订单"
            }
        }

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """处理家政员相关查询"""
        customer_query = state["customer_query"]
        session_id = state.get("session_id", "default")

        # 对话轮次由 classify / 外层节点写入 persisted_dialogue
        conversation_context = self._get_conversation_context(session_id, state)

        # 按路由结果决定：RAG、动态工具，或两者都做
        matched_info = self._build_agent_context(
            customer_query, state, self._match_housekeeper_info
        )

        # 构建系统提示并增强对话上下文说明
        base_system_prompt = f"""你是{self.name}，专门负责{self.role}。
        你的专业领域包括：{', '.join(self.expertise)}

        请根据客户需求提供家政员选择和匹配建议：
        - 先了解客户要做的服务类型、房屋情况、预算和上门时间
        - 说明不同等级家政员的差异，帮助客户按需求选择
        - 涉及具体家政员的评分、价格、档期时，提醒以家政员详情页和预约页实时信息为准
        - 客户询问某位家政员是否可约时，不要编造档期，引导客户在预约页选择日期查看
        - 可约时段只包含「明天起 3 个自然日」（明天、后天、大后天），不支持当天预约；回答时不要提今天或第 4 天及以后的档期

        回答要实用、有针对性。如果知识库中没有对应信息，请引导客户查看家政员详情页或转人工客服。"""

        system_prompt = self._enhance_system_prompt_with_context(base_system_prompt)

        # 构建消息列表
        messages = []

        # 添加对话历史上下文（如果有的话）
        if conversation_context:
            context_message = f"""对话历史上下文：
{conversation_context}

请基于以上对话历史和当前查询，提供连贯的建议。"""
            messages.append(SystemMessage(content=context_message))

        # 添加系统提示
        messages.append(SystemMessage(content=system_prompt))

        # 如果有匹配的家政员信息，添加到上下文中
        if matched_info:
            housekeeper_context = f"""家政员资料与规则：
{matched_info}

当前查询：{customer_query}"""
            messages.append(HumanMessage(content=housekeeper_context))
        else:
            messages.append(HumanMessage(content=customer_query))

        # 调用LLM（流式，token 会实时推送给前端）
        try:
            response_content = self._invoke_llm(messages)
        except Exception as e:
            print(f"家政员顾问调用LLM时出错: {e}")
            response_content = "抱歉，处理您关于家政员的咨询时遇到技术问题，请稍后重试。"

        state["response"] = response_content
        state["current_agent"] = self.name
        state["tools_used"].append(f"{self.name}_processing")

        return state

    def _match_housekeeper_info(self, query: str) -> str:
        """匹配查询中的家政员信息"""
        query_lower = query.lower()
        matched_info: List[str] = []

        # 1. 按分类名命中
        for category, details in self.housekeeper_database.items():
            if category in query_lower:
                info_text = f"【{category}】\n"
                for key, value in details.items():
                    info_text += f"• {key}：{value}\n"
                matched_info.append(info_text)

        # 2. 按关键词命中对应分类
        keyword_map = {
            "家政员等级": ["金牌", "银牌", "铜牌", "等级", "级别", "无牌"],
            "筛选维度": ["评分", "评价", "口碑", "价格", "单价", "经验", "健康证", "证书", "资质", "擅长"],
            "可预约时段": ["预约", "档期", "时间", "有空", "可约", "改约", "约满", "周末"],
            "选择建议": ["推荐", "怎么选", "哪个好", "适合", "选谁", "挑"]
        }
        for category, keywords in keyword_map.items():
            if category in [info.split("】")[0].replace("【", "") for info in matched_info]:
                continue
            if any(k in query_lower for k in keywords):
                info_text = f"【{category}】\n"
                for key, value in self.housekeeper_database[category].items():
                    info_text += f"• {key}：{value}\n"
                matched_info.append(info_text)

        # 3. 没有命中时给出总览
        if not matched_info:
            overview = []
            for category, details in self.housekeeper_database.items():
                overview.append(f"• {category}：{', '.join(details.keys())}")
            matched_info.append("【家政员信息总览】\n" + "\n".join(overview))

        return "\n".join(matched_info)
