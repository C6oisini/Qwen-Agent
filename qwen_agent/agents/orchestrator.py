# Copyright 2023 The Qwen team, Alibaba Group. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Orchestrator Agent - 支持 Agent 间交互的协调器

功能:
1. 协调多个 Agent 协同工作
2. 管理 Agent 间的消息传递
3. 支持任务分解和结果汇总
4. 维护共享上下文，支持 Agent 间信息共享
"""

import copy
import json
from typing import Dict, Iterator, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime

from qwen_agent import Agent, MultiAgentHub
from qwen_agent.agents.assistant import Assistant
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import ASSISTANT, ROLE, SYSTEM, USER, Message
from qwen_agent.log import logger
from qwen_agent.tools import BaseTool


@dataclass
class AgentMessage:
    """Agent 间传递的消息"""
    from_agent: str
    to_agent: str  # 可以是 "all" 表示广播
    content: str
    message_type: str = "info"  # info, request, response, alert
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict = field(default_factory=dict)


@dataclass
class SharedContext:
    """多 Agent 共享的上下文"""
    # 最近的分析结果
    latest_analysis: Dict = field(default_factory=dict)
    # 当前执行的任务
    current_tasks: List[Dict] = field(default_factory=list)
    # 历史决策记录
    decision_history: List[Dict] = field(default_factory=list)
    # Agent 间消息队列
    message_queue: List[AgentMessage] = field(default_factory=list)
    # 全局状态
    global_state: Dict = field(default_factory=dict)

    def add_message(self, msg: AgentMessage):
        self.message_queue.append(msg)

    def get_messages_for(self, agent_name: str) -> List[AgentMessage]:
        """获取发给特定 agent 的消息"""
        return [m for m in self.message_queue
                if m.to_agent == agent_name or m.to_agent == "all"]

    def clear_messages_for(self, agent_name: str):
        """清除已处理的消息"""
        self.message_queue = [m for m in self.message_queue
                              if m.to_agent != agent_name and m.to_agent != "all"]


ORCHESTRATOR_PROMPT = '''你是一个多 Agent 系统的协调者。你负责分析用户请求，分解任务，并协调多个专家 Agent 协同工作。

你管理的 Agent 团队:
{agent_descs}

## 你的职责:
1. **任务分析**: 理解用户需求，判断需要哪些 Agent 参与
2. **任务分解**: 将复杂任务拆分为子任务
3. **协调执行**: 按顺序或并行调用相关 Agent
4. **信息传递**: 将一个 Agent 的输出传递给另一个 Agent
5. **结果汇总**: 整合多个 Agent 的结果，给出最终答案

## 输出格式:
当你需要调用 Agent 时，使用以下格式:

【计划】
描述你的执行计划

【调用】Agent名称
给这个Agent的具体指令

【传递】从Agent1 → Agent2
需要传递的信息摘要

【汇总】
最终结果汇总

## 重要规则:
- 分析类任务优先交给分析助手
- 投放/创建类任务交给投放助手
- 项目调整类任务交给项目管理助手
- 素材处理交给 RAG 助手
- Agent 之间可以传递信息，但你需要明确指出传递什么
- 如果一个 Agent 的结果需要另一个 Agent 处理，你要协调这个流程
'''


class Orchestrator(Assistant, MultiAgentHub):
    """协调多个 Agent 协同工作的编排器"""

    def __init__(self,
                 agents: List[Agent],
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 name: str = "协调者",
                 description: str = "协调多个专家 Agent 协同工作",
                 **kwargs):

        self._agents = agents
        self.shared_context = SharedContext()

        agent_descs = '\n'.join([f'- {a.name}: {a.description}' for a in agents])
        # Use custom system_message from kwargs if provided, otherwise use default
        custom_system_message = kwargs.pop('system_message', None)
        system_message = custom_system_message or ORCHESTRATOR_PROMPT.format(agent_descs=agent_descs)

        super().__init__(
            function_list=function_list,
            llm=llm,
            system_message=system_message,
            name=name,
            description=description,
            **kwargs
        )

    def _run(self, messages: List[Message], lang: str = 'zh', **kwargs) -> Iterator[List[Message]]:
        """执行协调流程"""
        messages = copy.deepcopy(messages)

        # 直接使用用户原始消息，不添加额外指令
        plan_response = []
        for resp in super()._run(messages=messages, lang=lang, **kwargs):
            plan_response = resp
            yield resp

        if not plan_response:
            print("[Orchestrator] No plan response received")
            return

        # 提取 content（可能是字符串或列表）
        plan_content = self._extract_content(plan_response[-1])
        print(f"[Orchestrator] Plan content length: {len(plan_content)}")
        print(f"[Orchestrator] Plan content preview: {plan_content[:500]}...")

        # 2. 解析计划中的 Agent 调用
        agent_calls = self._parse_agent_calls(plan_content)
        print(f"[Orchestrator] Parsed {len(agent_calls)} agent calls")
        for i, call in enumerate(agent_calls):
            print(f"[Orchestrator]   Call {i+1}: agent='{call.get('agent')}', instruction_len={len(call.get('instruction', ''))}")

        if not agent_calls:
            # 没有解析到 agent 调用，直接返回
            print("[Orchestrator] No agent calls found in plan, returning")
            return

        # 3. 执行 Agent 调用
        all_responses = list(plan_response)  # 复制而不是引用
        context_for_next = ""

        for call in agent_calls:
            agent_name = call.get('agent')
            instruction = call.get('instruction', '')

            print(f"[Orchestrator] Looking for agent: '{agent_name}'")
            print(f"[Orchestrator] Available agents: {[a.name for a in self.agents]}")

            agent = self._get_agent_by_name(agent_name)
            if not agent:
                print(f"[Orchestrator] ❌ Agent not found: {agent_name}")
                continue

            print(f"[Orchestrator] ✓ Found agent: {agent.name}")

            # 构建给子 Agent 的消息
            # 从原始消息中提取用户请求
            original_request = ""
            for msg in messages:
                if msg.role == USER:
                    original_request = self._extract_content(msg)
                    break

            # 构建给子 Agent 的消息，包含上下文
            agent_messages = []
            if context_for_next:
                agent_messages.append(Message(
                    role=USER,
                    content=f"【原始请求】\n{original_request}\n\n【来自协调者的上下文】\n{context_for_next}\n\n【当前任务】\n{instruction}"
                ))
            else:
                agent_messages.append(Message(role=USER, content=f"【原始请求】\n{original_request}\n\n【当前任务】\n{instruction}"))

            # 调用子 Agent
            print(f"[Orchestrator] 🚀 Calling agent: {agent.name}")
            agent_response = []
            for resp in agent.run(messages=agent_messages, lang=lang, **kwargs):
                agent_response = resp

            print(f"[Orchestrator] Agent {agent.name} returned {len(agent_response) if agent_response else 0} messages")

            if agent_response:
                # 标记响应来自哪个 Agent
                for r in agent_response:
                    if hasattr(r, 'role') and r.role == ASSISTANT:
                        r.name = agent.name
                    elif isinstance(r, dict) and r.get('role') == ASSISTANT:
                        r['name'] = agent.name
                all_responses.extend(agent_response)

                # 更新上下文用于下一个 Agent
                last_content = self._extract_content(agent_response[-1])
                context_for_next = f"【{agent.name}的结果】\n{last_content}"

                # 更新共享上下文
                self.shared_context.latest_analysis[agent.name] = last_content

                yield all_responses

        # 4. 如果有多个 Agent 参与，让协调者汇总结果
        if len(agent_calls) > 1:
            summary_prompt = self._build_summary_prompt(agent_calls, all_responses)
            summary_messages = messages + [Message(role=USER, content=summary_prompt)]

            for resp in super()._run(messages=summary_messages, lang=lang, **kwargs):
                all_responses.extend(resp)
                yield all_responses

    def _extract_content(self, message) -> str:
        """从 Message 中提取文本内容"""
        if isinstance(message, dict):
            content = message.get('content', '')
        else:
            content = getattr(message, 'content', '')

        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            # multimodal content
            texts = []
            for item in content:
                if isinstance(item, dict) and 'text' in item:
                    texts.append(item['text'])
                elif hasattr(item, 'text') and item.text:
                    texts.append(item.text)
            return '\n'.join(texts)
        return str(content)

    def _parse_agent_calls(self, content: str) -> List[Dict]:
        """解析协调者输出中的 Agent 调用"""
        calls = []
        lines = content.split('\n')

        current_agent = None
        current_instruction = []

        for line in lines:
            if line.startswith('【调用】'):
                # 保存上一个调用
                if current_agent:
                    calls.append({
                        'agent': current_agent,
                        'instruction': '\n'.join(current_instruction)
                    })
                # 开始新的调用
                current_agent = line.replace('【调用】', '').strip()
                current_instruction = []
            elif current_agent and not line.startswith('【'):
                current_instruction.append(line)

        # 保存最后一个调用
        if current_agent:
            calls.append({
                'agent': current_agent,
                'instruction': '\n'.join(current_instruction)
            })

        return calls

    def _get_agent_by_name(self, name: str) -> Optional[Agent]:
        """根据名称获取 Agent"""
        for agent in self.agents:
            if agent.name == name or name in agent.name:
                return agent
        return None

    def _build_summary_prompt(self, calls: List[Dict], responses: List[Message]) -> str:
        """构建汇总提示"""
        summary = "请汇总以下各个 Agent 的执行结果，给出最终答案:\n\n"

        for resp in responses:
            resp_role = resp.role if hasattr(resp, 'role') else resp.get('role', '')
            resp_name = resp.name if hasattr(resp, 'name') else resp.get('name', '')
            if resp_role == ASSISTANT and resp_name:
                content = self._extract_content(resp)
                summary += f"【{resp_name}】\n{content}\n\n"

        summary += "请综合以上信息，给出完整的汇总结论。"
        return summary

    def send_message(self, from_agent: str, to_agent: str, content: str,
                     message_type: str = "info", metadata: Dict = None):
        """Agent 间发送消息"""
        msg = AgentMessage(
            from_agent=from_agent,
            to_agent=to_agent,
            content=content,
            message_type=message_type,
            metadata=metadata or {}
        )
        self.shared_context.add_message(msg)
        logger.info(f"Message sent: {from_agent} -> {to_agent}: {content[:50]}...")

    def broadcast(self, from_agent: str, content: str, message_type: str = "info"):
        """广播消息给所有 Agent"""
        self.send_message(from_agent, "all", content, message_type)
