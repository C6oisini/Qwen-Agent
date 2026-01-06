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

import os
import pprint
import re
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

from qwen_agent import Agent, MultiAgentHub
from qwen_agent.agents.user_agent import PENDING_USER_INPUT
from qwen_agent.gui.gradio_utils import format_cover_html
from qwen_agent.gui.utils import convert_fncall_to_text, convert_history_to_chatbot, get_avatar_image
from qwen_agent.llm.schema import AUDIO, CONTENT, FILE, IMAGE, NAME, ROLE, SYSTEM, USER, VIDEO, Message
from qwen_agent.log import logger
from qwen_agent.utils.utils import print_traceback


class WebUI:
    """A Common chatbot application for agent."""

    def __init__(self, agent: Union[Agent, MultiAgentHub, List[Agent]], chatbot_config: Optional[dict] = None):
        """
        Initialization the chatbot.

        Args:
            agent: The agent or a list of agents,
                supports various types of agents such as Assistant, GroupChat, Router, etc.
            chatbot_config: The chatbot configuration.
                Set the configuration as {'user.name': '', 'user.avatar': '', 'agent.avatar': '', 'input.placeholder': '', 'prompt.suggestions': [],
                'history.max_length': 20, 'history.session_dir': './sessions'}.
        """
        chatbot_config = chatbot_config or {}

        # 历史管理配置
        self.max_history_length = chatbot_config.get('history.max_length', 20)
        self.session_dir = Path(chatbot_config.get('history.session_dir', './sessions'))
        self.session_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(agent, MultiAgentHub):
            self.agent_list = [a for a in agent.nonuser_agents]
            self.agent_hub = agent
        elif isinstance(agent, list):
            self.agent_list = agent
            self.agent_hub = None
        else:
            self.agent_list = [agent]
            self.agent_hub = None

        # Fallback if agent_list is empty but we have an agent_hub
        if not self.agent_list and self.agent_hub:
            logger.warning('nonuser_agents is empty, using agent_hub as fallback')
            self.agent_list = [self.agent_hub]

        user_name = chatbot_config.get('user.name', 'user')
        self.user_config = {
            'name': user_name,
            'avatar': chatbot_config.get(
                'user.avatar',
                get_avatar_image(user_name),
            ),
        }

        self.agent_config_list = [{
            'name': a.name,
            'avatar': chatbot_config.get(
                'agent.avatar',
                get_avatar_image(a.name),
            ),
            'description': a.description or "I'm a helpful assistant.",
        } for a in self.agent_list]

        self.input_placeholder = chatbot_config.get('input.placeholder', '跟我聊聊吧～')
        self.prompt_suggestions = chatbot_config.get('prompt.suggestions', [])
        self.verbose = chatbot_config.get('verbose', False)

    """
    Run the chatbot.

    Args:
        messages: The chat history.
    """

    def run(self,
            messages: List[Message] = None,
            share: bool = False,
            server_name: str = None,
            server_port: int = None,
            concurrency_limit: int = 10,
            enable_mention: bool = False,
            **kwargs):
        self.run_kwargs = kwargs

        from qwen_agent.gui.gradio_dep import gr, mgr, ms

        customTheme = gr.themes.Default(
            primary_hue=gr.themes.utils.colors.blue,
            radius_size=gr.themes.utils.sizes.radius_none,
        )

        with gr.Blocks(
                css=os.path.join(os.path.dirname(__file__), 'assets/appBot.css'),
                theme=customTheme,
        ) as demo:
            history = gr.State([])
            with ms.Application():
                with gr.Row(elem_classes='container'):
                    with gr.Column(scale=4):
                        chatbot = mgr.Chatbot(value=convert_history_to_chatbot(messages=messages),
                                              avatar_images=[
                                                  self.user_config,
                                                  self.agent_config_list,
                                              ],
                                              height=850,
                                              avatar_image_width=80,
                                              flushing=False,
                                              show_copy_button=True,
                                              latex_delimiters=[{
                                                  'left': '\\(',
                                                  'right': '\\)',
                                                  'display': True
                                              }, {
                                                  'left': '\\begin{equation}',
                                                  'right': '\\end{equation}',
                                                  'display': True
                                              }, {
                                                  'left': '\\begin{align}',
                                                  'right': '\\end{align}',
                                                  'display': True
                                              }, {
                                                  'left': '\\begin{alignat}',
                                                  'right': '\\end{alignat}',
                                                  'display': True
                                              }, {
                                                  'left': '\\begin{gather}',
                                                  'right': '\\end{gather}',
                                                  'display': True
                                              }, {
                                                  'left': '\\begin{CD}',
                                                  'right': '\\end{CD}',
                                                  'display': True
                                              }, {
                                                  'left': '\\[',
                                                  'right': '\\]',
                                                  'display': True
                                              }])

                        with gr.Row():
                            input = mgr.MultimodalInput(placeholder=self.input_placeholder, scale=8)
                            stop_btn = gr.Button("⏹️ 停止", variant="stop", scale=1, visible=False)

                        audio_input = gr.Audio(
                            sources=["microphone"],
                            type="filepath"
                        )

                    with gr.Column(scale=1):
                        if len(self.agent_list) > 1:
                            agent_selector = gr.Dropdown(
                                [(agent.name, i) for i, agent in enumerate(self.agent_list)],
                                label='Agents',
                                info='选择一个Agent',
                                value=0,
                                interactive=True,
                            )

                        agent_info_block = self._create_agent_info_block()

                        # 插件面板（可折叠）
                        with gr.Accordion("插件列表", open=False) as plugins_accordion:
                            agent_plugins_block = self._create_agent_plugins_block()

                        # 会话管理面板
                        with gr.Accordion("会话管理", open=False):
                            with gr.Row():
                                session_name_input = gr.Textbox(
                                    label="会话名称",
                                    placeholder="留空则使用时间戳",
                                    scale=3
                                )
                                save_btn = gr.Button("💾 保存", scale=1, size="sm")

                            session_dropdown = gr.Dropdown(
                                label="加载已保存的会话",
                                choices=self.list_sessions(),
                                interactive=True,
                            )

                            with gr.Row():
                                load_btn = gr.Button("📂 加载会话", variant="secondary", size="sm")
                                refresh_btn = gr.Button("🔄 刷新列表", size="sm")
                                clear_btn = gr.Button("🗑️ 清空历史", variant="stop", size="sm")

                            session_info = gr.Textbox(
                                label="状态信息",
                                interactive=False,
                                max_lines=2
                            )

                        if self.prompt_suggestions:
                            gr.Examples(
                                label='推荐对话',
                                examples=self.prompt_suggestions,
                                inputs=[input],
                            )

                    if len(self.agent_list) > 1:
                        agent_selector.change(
                            fn=self.change_agent,
                            inputs=[agent_selector],
                            outputs=[agent_selector, agent_info_block, agent_plugins_block],
                            queue=False,
                        )

                    input_promise = input.submit(
                        fn=self.add_text,
                        inputs=[input, audio_input, chatbot, history],
                        outputs=[input, audio_input, chatbot, history],
                        queue=False,
                    ).then(
                        fn=lambda: gr.update(visible=True),
                        inputs=None,
                        outputs=[stop_btn],
                        queue=False,
                    )

                    if len(self.agent_list) > 1 and enable_mention:
                        agent_run_event = input_promise.then(
                            self.add_mention,
                            [chatbot, agent_selector],
                            [chatbot, agent_selector],
                        ).then(
                            self.agent_run,
                            [chatbot, history, agent_selector],
                            [chatbot, history, agent_selector],
                        )
                    else:
                        agent_run_event = input_promise.then(
                            self.agent_run,
                            [chatbot, history],
                            [chatbot, history],
                        )

                    agent_run_event.then(self.flushed, None, [input]).then(
                        fn=lambda: gr.update(visible=False),
                        inputs=None,
                        outputs=[stop_btn],
                        queue=False,
                    )

                    # 停止按钮事件
                    stop_btn.click(
                        fn=lambda: gr.update(visible=False),
                        inputs=None,
                        outputs=[stop_btn],
                        queue=False,
                        cancels=[agent_run_event],
                    )

                    # 会话管理事件绑定
                    save_btn.click(
                        fn=self.save_session_wrapper,
                        inputs=[history, session_name_input],
                        outputs=[session_info, session_dropdown],
                        queue=False,
                    )

                    load_btn.click(
                        fn=self.load_session_wrapper,
                        inputs=[session_dropdown],
                        outputs=[chatbot, history, session_info],
                        queue=False,
                    )

                    refresh_btn.click(
                        fn=self.refresh_sessions,
                        inputs=[],
                        outputs=[session_dropdown, session_info],
                        queue=False,
                    )

                    clear_btn.click(
                        fn=self.clear_history,
                        inputs=[chatbot, history],
                        outputs=[chatbot, history, session_info],
                        queue=False,
                    )

            demo.load(None)

        demo.queue(default_concurrency_limit=concurrency_limit).launch(share=share,
                                                                       server_name=server_name,
                                                                       server_port=server_port)

    def change_agent(self, agent_selector):
        yield agent_selector, self._create_agent_info_block(agent_selector), self._create_agent_plugins_block(
            agent_selector)

    def add_text(self, _input, _audio_input, _chatbot, _history):
        _history.append({
            ROLE: USER,
            CONTENT: [{
                'text': _input.text
            }],
        })

        if self.user_config[NAME]:
            _history[-1][NAME] = self.user_config[NAME]
        
        # if got audio from microphone, append it to the multimodal inputs
        if _audio_input:
            from qwen_agent.gui.gradio_dep import gr, mgr, ms
            audio_input_file = gr.data_classes.FileData(path=_audio_input, mime_type="audio/wav")
            _input.files.append(audio_input_file)

        if _input.files:
            for file in _input.files:
                if file.mime_type.startswith('image/'):
                    _history[-1][CONTENT].append({IMAGE: 'file://' + file.path})
                elif file.mime_type.startswith('audio/'):
                    _history[-1][CONTENT].append({AUDIO: 'file://' + file.path})
                elif file.mime_type.startswith('video/'):
                    _history[-1][CONTENT].append({VIDEO: 'file://' + file.path})
                else:
                    _history[-1][CONTENT].append({FILE: file.path})

        _chatbot.append([_input, None])

        from qwen_agent.gui.gradio_dep import gr

        yield gr.update(interactive=False, value=None), None, _chatbot, _history

    def add_mention(self, _chatbot, _agent_selector):
        if len(self.agent_list) == 1:
            yield _chatbot, _agent_selector

        query = _chatbot[-1][0].text
        match = re.search(r'@\w+\b', query)
        if match:
            _agent_selector = self._get_agent_index_by_name(match.group()[1:])

        agent_name = self.agent_list[_agent_selector].name

        if ('@' + agent_name) not in query and self.agent_hub is None:
            _chatbot[-1][0].text = '@' + agent_name + ' ' + query

        yield _chatbot, _agent_selector

    def agent_run(self, _chatbot, _history, _agent_selector=None):
        if self.verbose:
            logger.info('agent_run input:\n' + pprint.pformat(_history, indent=2))

        # Guard against empty chatbot state
        if not _chatbot:
            if _agent_selector is not None:
                yield _chatbot, _history, _agent_selector
            else:
                yield _chatbot, _history
            return

        # Guard against empty agent list
        if not self.agent_list:
            logger.warning('agent_list is empty, cannot process agent_run')
            if _agent_selector is not None:
                yield _chatbot, _history, _agent_selector
            else:
                yield _chatbot, _history
            return

        num_input_bubbles = len(_chatbot) - 1
        num_output_bubbles = 1
        _chatbot[-1][1] = [None for _ in range(len(self.agent_list))]

        agent_runner = self.agent_list[_agent_selector or 0] if self.agent_list else None
        if agent_runner is None and self.agent_hub is None:
            if _agent_selector is not None:
                yield _chatbot, _history, _agent_selector
            else:
                yield _chatbot, _history
            return
        if self.agent_hub:
            agent_runner = self.agent_hub

        # 应用滑动窗口机制：截断历史记录
        truncated_history = self.truncate_history(_history)
        if len(truncated_history) < len(_history):
            logger.info(f'History truncated: {len(_history)} -> {len(truncated_history)} messages')

        responses = []
        for responses in agent_runner.run(truncated_history, **self.run_kwargs):
            if not responses:
                continue
            if responses[-1][CONTENT] == PENDING_USER_INPUT:
                logger.info('Interrupted. Waiting for user input!')
                break

            display_responses = convert_fncall_to_text(responses)
            if not display_responses:
                continue
            if display_responses[-1][CONTENT] is None:
                continue

            while len(display_responses) > num_output_bubbles:
                # Create a new chat bubble
                _chatbot.append([None, None])
                _chatbot[-1][1] = [None for _ in range(len(self.agent_list))]
                num_output_bubbles += 1

            assert num_output_bubbles == len(display_responses)
            assert num_input_bubbles + num_output_bubbles == len(_chatbot)

            for i, rsp in enumerate(display_responses):
                agent_index = self._get_agent_index_by_name(rsp[NAME])
                _chatbot[num_input_bubbles + i][1][agent_index] = rsp[CONTENT]

            if len(self.agent_list) > 1:
                _agent_selector = agent_index

            if _agent_selector is not None:
                yield _chatbot, _history, _agent_selector
            else:
                yield _chatbot, _history

        if responses:
            _history.extend([res for res in responses if res[CONTENT] != PENDING_USER_INPUT])

        if _agent_selector is not None:
            yield _chatbot, _history, _agent_selector
        else:
            yield _chatbot, _history

        if self.verbose:
            logger.info('agent_run response:\n' + pprint.pformat(responses, indent=2))

    def flushed(self):
        from qwen_agent.gui.gradio_dep import gr

        return gr.update(interactive=True)

    def _get_agent_index_by_name(self, agent_name):
        if agent_name is None:
            return 0

        try:
            agent_name = agent_name.strip()
            for i, agent in enumerate(self.agent_list):
                if agent.name == agent_name:
                    return i
            return 0
        except Exception:
            print_traceback()
            return 0

    def _create_agent_info_block(self, agent_index=0):
        from qwen_agent.gui.gradio_dep import gr

        agent_config_interactive = self.agent_config_list[agent_index]

        return gr.HTML(
            format_cover_html(
                bot_name=agent_config_interactive['name'],
                bot_description=agent_config_interactive['description'],
                bot_avatar=agent_config_interactive['avatar'],
            ))

    def _create_agent_plugins_block(self, agent_index=0):
        from qwen_agent.gui.gradio_dep import gr

        agent_interactive = self.agent_list[agent_index]

        if agent_interactive.function_map:
            capabilities = [key for key in agent_interactive.function_map.keys()]
            return gr.CheckboxGroup(
                label='插件',
                value=capabilities,
                choices=capabilities,
                interactive=False,
            )

        else:
            return gr.CheckboxGroup(
                label='插件',
                value=[],
                choices=[],
                interactive=False,
            )

    # ==================== 历史管理方法 ====================

    def truncate_history(self, _history: List[dict]) -> List[dict]:
        """截断历史记录，只保留最近的消息

        Args:
            _history: 完整的历史消息列表

        Returns:
            截断后的历史消息列表
        """
        if len(_history) <= self.max_history_length:
            return self._ensure_starts_with_user(_history)

        # 保留系统消息
        system_messages = [msg for msg in _history if msg.get(ROLE) == SYSTEM]

        # 保留最近的消息
        recent_messages = _history[-self.max_history_length:]

        # 确保第一条非系统消息是用户消息（LLM 要求）
        recent_messages = self._ensure_starts_with_user(recent_messages)

        # 如果最近的消息中没有系统消息，则添加
        if system_messages and not any(msg.get(ROLE) == SYSTEM for msg in recent_messages):
            return system_messages + recent_messages

        return recent_messages

    def _ensure_starts_with_user(self, messages: List[dict]) -> List[dict]:
        """确保消息列表的第一条非系统消息是用户消息

        Args:
            messages: 消息列表

        Returns:
            处理后的消息列表
        """
        if not messages:
            return messages

        # 找到第一条用户消息的位置
        first_user_idx = None
        for i, msg in enumerate(messages):
            if msg.get(ROLE) == USER:
                first_user_idx = i
                break

        if first_user_idx is None:
            # 没有用户消息，返回空列表（保留系统消息）
            return [msg for msg in messages if msg.get(ROLE) == SYSTEM]

        if first_user_idx == 0:
            return messages

        # 保留第一条用户消息之前的系统消息
        system_before = [msg for msg in messages[:first_user_idx] if msg.get(ROLE) == SYSTEM]
        return system_before + messages[first_user_idx:]

    def save_session(self, _history: List[dict], session_name: str = None) -> str:
        """保存会话到文件

        Args:
            _history: 历史消息列表
            session_name: 会话名称，如果为空则使用时间戳

        Returns:
            会话文件名
        """
        if not session_name:
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 确保文件名安全
        session_name = re.sub(r'[^\w\-]', '_', session_name)
        session_file = self.session_dir / f"{session_name}.json"

        try:
            # 转换 Message 对象为字典
            serializable_history = []
            for msg in _history:
                if isinstance(msg, Message):
                    serializable_history.append(msg.model_dump())
                elif isinstance(msg, dict):
                    serializable_history.append(msg)
                else:
                    serializable_history.append(dict(msg))

            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'saved_at': datetime.now().isoformat(),
                    'message_count': len(serializable_history),
                    'history': serializable_history
                }, f, ensure_ascii=False, indent=2)

            logger.info(f"Session saved to {session_file}")
            return session_name
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            print_traceback()
            return ""

    def load_session(self, session_name: str) -> List[dict]:
        """从文件加载会话

        Args:
            session_name: 会话名称（不含 .json 后缀）

        Returns:
            历史消息列表，如果加载失败则返回空列表
        """
        session_file = self.session_dir / f"{session_name}.json"

        if not session_file.exists():
            logger.warning(f"Session file not found: {session_file}")
            return []

        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            logger.info(f"Session loaded from {session_file}, {data.get('message_count', 0)} messages")
            return data.get('history', [])
        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            print_traceback()
            return []

    def list_sessions(self) -> List[str]:
        """列出所有可用的会话

        Returns:
            会话名称列表（不含 .json 后缀）
        """
        try:
            sessions = [f.stem for f in self.session_dir.glob("*.json")]
            return sorted(sessions, reverse=True)  # 按时间倒序
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []

    def clear_history(self, _chatbot, _history):
        """清空历史记录

        Returns:
            空的 chatbot 和 history
        """
        logger.info("History cleared")
        return [], [], "历史记录已清空"

    def save_session_wrapper(self, _history, session_name):
        """保存会话的包装函数，用于 Gradio 事件

        Returns:
            更新后的 session_info
        """
        from qwen_agent.gui.gradio_dep import gr

        if not _history:
            return "没有历史记录可保存", gr.update()

        saved_name = self.save_session(_history, session_name or None)
        if saved_name:
            return f"✅ 会话已保存: {saved_name}", gr.update(choices=self.list_sessions())
        else:
            return "❌ 保存失败，请查看日志", gr.update()

    def load_session_wrapper(self, session_name):
        """加载会话的包装函数，用于 Gradio 事件

        Returns:
            更新后的 chatbot, history, session_info
        """
        from qwen_agent.gui.gradio_dep import gr

        if not session_name:
            return gr.update(), [], "请先选择一个会话"

        loaded_history = self.load_session(session_name)
        if loaded_history:
            chatbot = convert_history_to_chatbot(messages=loaded_history)
            return chatbot, loaded_history, f"✅ 已加载会话: {session_name}"
        else:
            return gr.update(), [], "❌ 加载失败，请查看日志"

    def refresh_sessions(self):
        """刷新会话列表

        Returns:
            更新后的 dropdown choices
        """
        from qwen_agent.gui.gradio_dep import gr

        sessions = self.list_sessions()
        return gr.update(choices=sessions), f"✅ 已刷新，共 {len(sessions)} 个会话"
