"""
平台与通用客服智能体
专门负责平台信息、账号使用和小程序常见问题
"""

from typing import Dict, List, Any
from langchain_core.messages import HumanMessage, SystemMessage
from .base_agent import BaseAgent


class GeneralAgent(BaseAgent):
    agent_key = "general_agent"

    def __init__(self):
        super().__init__(
            name="平台客服",
            role="平台信息、账号使用与小程序问题处理",
            expertise=["平台介绍", "服务范围与时间", "账号登录", "地址管理", "消息通知", "小程序使用"]
        )

        # TODO: 正式版应从 about / updates / service_intro 表读取平台信息
        # 这里整理平台通用信息和使用指引，具体以小程序页面展示为准
        self.platform_database = {
            "平台信息": {
                "平台介绍": "智能家政服务平台，提供日常保洁、家电清洗、维修服务、月嫂育儿、老人陪护、空气治理等服务",
                "服务范围": "具体可服务的城市和区域以首页及预约页展示为准，偏远区域可能暂未开通",
                "服务时间": "在线客服 7×24 小时；上门服务时间以预约页可选时段为准",
                "联系方式": "可在小程序『我的-关于我们』查看客服电话和在线客服入口",
                "关于我们": "在『我的-关于我们』查看平台介绍、服务保障和联系方式",
                "最新动态": "在首页『动态』或『服务介绍』中查看平台公告和服务资讯"
            },
            "账号与使用": {
                "登录": "进入小程序后按提示完成微信授权登录；登录失败可检查网络、微信授权状态，或退出小程序重新进入",
                "退出登录": "在『我的』页面底部点击退出登录",
                "地址管理": "进入『我的-地址管理』，可新增、编辑、删除地址并设置默认地址",
                "预约清单": "底部『预约清单』可查看已加入的服务，确认后提交订单",
                "历史订单": "进入『我的-历史订单』查看全部订单及状态",
                "我的优惠券": "进入『我的-我的优惠券』查看未使用、已使用和已过期的券",
                "消息中心": "进入『我的-消息中心』查看接单、开始服务、完成、取消、退款等通知",
                "家政员工作台": "家政员可从『我的-家政员工作台』登录，处理接单和排班"
            },
            "小程序常见问题": {
                "定位失败": "检查微信定位权限是否开启，或在地址管理中手动填写地址",
                "收不到通知": "检查微信通知权限和小程序订阅消息设置，未开启可在消息中心查看历史通知",
                "页面白屏或卡顿": "退出小程序重新进入，或清理微信缓存后重试",
                "支付无法唤起": "检查网络和微信支付权限，可尝试重新下单或更换支付方式",
                "图片不显示": "检查网络连接，切换网络后重新进入页面"
            },
            "服务保障": {
                "身份核验": "家政员入驻需经过身份和资质审核",
                "健康证": "家政员详情页可查看健康证等资质信息",
                "服务评价": "服务完成后可对家政员进行评价，评价会展示在家政员详情页",
                "隐私保护": "平台仅在提供服务所需范围内使用您的地址和联系方式",
                "交易安全": "请通过平台下单和支付，不要与家政员私下转账交易"
            }
        }

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """处理平台与通用咨询"""
        customer_query = state["customer_query"]
        session_id = state.get("session_id", "default")

        # 对话轮次由 classify / 外层节点写入 persisted_dialogue
        conversation_context = self._get_conversation_context(session_id, state)

        # 按路由结果决定：RAG、动态工具，或两者都做
        matched_info = self._build_agent_context(
            customer_query, state, self._match_platform_info
        )

        # 构建系统提示并增强对话上下文说明
        base_system_prompt = f"""你是{self.name}，专门负责{self.role}。
        你的专业领域包括：{', '.join(self.expertise)}

        请友好、简洁地解答平台使用类问题：
        - 优先告诉客户在哪个页面、点哪个入口能解决问题
        - 小程序操作问题给出具体步骤，避免只给笼统建议
        - 平台介绍、服务范围、联系方式等信息以小程序页面公示为准
        - 如果问题实际属于服务、订单、支付或投诉，礼貌说明并引导客户描述具体诉求

        回答要像真人客服，不要机械罗列。无法确认的信息不要编造。"""

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

        # 如果有匹配的平台信息，添加到上下文中
        if matched_info:
            platform_context = f"""平台信息与使用指引：
{matched_info}

当前查询：{customer_query}"""
            messages.append(HumanMessage(content=platform_context))
        else:
            messages.append(HumanMessage(content=customer_query))

        # 调用LLM（流式，token 会实时推送给前端）
        try:
            response_content = self._invoke_llm(messages)
        except Exception as e:
            print(f"平台客服调用LLM时出错: {e}")
            response_content = "抱歉，处理您的咨询时遇到技术问题，请稍后重试。"

        state["response"] = response_content
        state["current_agent"] = self.name
        state["tools_used"].append(f"{self.name}_processing")

        return state

    def _match_platform_info(self, query: str) -> str:
        """匹配查询中的平台信息"""
        query_lower = query.lower()
        matched_info: List[str] = []

        # 1. 按分类名命中
        for category, details in self.platform_database.items():
            if category in query_lower:
                info_text = f"【{category}】\n"
                for key, value in details.items():
                    info_text += f"• {key}：{value}\n"
                matched_info.append(info_text)

        # 2. 按关键词命中对应分类
        keyword_map = {
            "平台信息": ["你们是", "平台", "公司", "关于我们", "服务范围", "哪些城市", "营业时间",
                         "客服电话", "联系方式", "动态", "公告"],
            "账号与使用": ["登录", "退出", "账号", "地址管理", "默认地址", "预约清单", "历史订单",
                           "我的优惠券", "消息中心", "工作台", "在哪看", "怎么改"],
            "小程序常见问题": ["定位", "通知", "白屏", "卡顿", "打不开", "闪退", "图片", "加载", "收不到"],
            "服务保障": ["保障", "安全", "健康证", "资质", "保险", "隐私", "评价", "靠谱吗"]
        }
        existing = [info.split("】")[0].replace("【", "") for info in matched_info]
        for category, keywords in keyword_map.items():
            if category in existing:
                continue
            if any(k in query_lower for k in keywords):
                info_text = f"【{category}】\n"
                for key, value in self.platform_database[category].items():
                    info_text += f"• {key}：{value}\n"
                matched_info.append(info_text)

        # 3. 没有命中时给出平台信息总览
        if not matched_info:
            info_text = "【平台信息总览】\n"
            for category, details in self.platform_database.items():
                info_text += f"• {category}：{', '.join(details.keys())}\n"
            matched_info.append(info_text)

        return "\n".join(matched_info)
