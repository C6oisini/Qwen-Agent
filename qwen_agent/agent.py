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

import copy
import json
import threading
import traceback
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional, Tuple, Union

from qwen_agent.llm import get_chat_model
from qwen_agent.llm.base import BaseChatModel
from qwen_agent.llm.schema import CONTENT, DEFAULT_SYSTEM_MESSAGE, ROLE, SYSTEM, ContentItem, Message
from qwen_agent.log import logger
from qwen_agent.tools import TOOL_REGISTRY, BaseTool, MCPManager
from qwen_agent.tools.base import ToolServiceError
from qwen_agent.tools.simple_doc_parser import DocParserError
from qwen_agent.utils.utils import has_chinese_messages, merge_generate_cfgs


class Agent(ABC):
    """A base class for Agent.

    An agent can receive messages and provide response by LLM or Tools.
    Different agents have distinct workflows for processing messages and generating responses in the `_run` method.
    """

    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[dict, BaseChatModel]] = None,
                 system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 **kwargs):
        """Initialization the agent.

        Args:
            function_list: One list of tool name, tool configuration or Tool object,
              such as 'code_interpreter', {'name': 'code_interpreter', 'timeout': 10}, or CodeInterpreter().
            llm: The LLM model configuration or LLM model object.
              Set the configuration as {'model': '', 'api_key': '', 'model_server': ''}.
            system_message: The specified system message for LLM chat.
            name: The name of this agent.
            description: The description of this agent, which will be used for multi_agent.
        """
        if isinstance(llm, dict):
            self.llm = get_chat_model(llm)
        else:
            self.llm = llm
        self.extra_generate_cfg: dict = {}

        self._function_map_lock = threading.RLock()  # Reentrant lock for thread-safe tool access
        self.function_map = {}
        if function_list:
            for tool in function_list:
                self._init_tool(tool)

        self.system_message = system_message
        self.name = name
        self.description = description

    def run_nonstream(self, messages: List[Union[Dict, Message]], **kwargs) -> Union[List[Message], List[Dict]]:
        """Same as self.run, but with stream=False,
        meaning it returns the complete response directly
        instead of streaming the response incrementally."""
        *_, last_responses = self.run(messages, **kwargs)
        return last_responses

    def run(self, messages: List[Union[Dict, Message]],
            **kwargs) -> Union[Iterator[List[Message]], Iterator[List[Dict]]]:
        """Return one response generator based on the received messages.

        This method performs a uniform type conversion for the inputted messages,
        and calls the _run method to generate a reply.

        Args:
            messages: A list of messages.

        Yields:
            The response generator.
        """
        messages = copy.deepcopy(messages)
        _return_message_type = 'dict'
        new_messages = []
        # Only return dict when all input messages are dict
        if not messages:
            _return_message_type = 'message'
        for msg in messages:
            if isinstance(msg, dict):
                new_messages.append(Message(**msg))
            else:
                new_messages.append(msg)
                _return_message_type = 'message'

        if 'lang' not in kwargs:
            if has_chinese_messages(new_messages):
                kwargs['lang'] = 'zh'
            else:
                kwargs['lang'] = 'en'

        if self.system_message:
            if not new_messages or new_messages[0][ROLE] != SYSTEM:
                # Add the system instruction to the agent
                new_messages.insert(0, Message(role=SYSTEM, content=self.system_message))
            else:
                # Already got system message in new_messages
                if isinstance(new_messages[0][CONTENT], str):
                    new_messages[0][CONTENT] = self.system_message + '\n\n' + new_messages[0][CONTENT]
                else:
                    assert isinstance(new_messages[0][CONTENT], list)
                    assert new_messages[0][CONTENT][0].text
                    new_messages[0][CONTENT] = [ContentItem(text=self.system_message + '\n\n')
                                               ] + new_messages[0][CONTENT]  # noqa

        for rsp in self._run(messages=new_messages, **kwargs):
            for i in range(len(rsp)):
                if not rsp[i].name and self.name:
                    rsp[i].name = self.name
            if _return_message_type == 'message':
                yield [Message(**x) if isinstance(x, dict) else x for x in rsp]
            else:
                yield [x.model_dump() if not isinstance(x, dict) else x for x in rsp]

    @abstractmethod
    def _run(self, messages: List[Message], lang: str = 'en', **kwargs) -> Iterator[List[Message]]:
        """Return one response generator based on the received messages.

        The workflow for an agent to generate a reply.
        Each agent subclass needs to implement this method.

        Args:
            messages: A list of messages.
            lang: Language, which will be used to select the language of the prompt
              during the agent's execution process.

        Yields:
            The response generator.
        """
        raise NotImplementedError

    def _call_llm(
        self,
        messages: List[Message],
        functions: Optional[List[Dict]] = None,
        stream: bool = True,
        extra_generate_cfg: Optional[dict] = None,
    ) -> Iterator[List[Message]]:
        """The interface of calling LLM for the agent.

        We prepend the system_message of this agent to the messages, and call LLM.

        Args:
            messages: A list of messages.
            functions: The list of functions provided to LLM.
            stream: LLM streaming output or non-streaming output.
              For consistency, we default to using streaming output across all agents.

        Yields:
            The response generator of LLM.
        """
        return self.llm.chat(messages=messages,
                             functions=functions,
                             stream=stream,
                             extra_generate_cfg=merge_generate_cfgs(
                                 base_generate_cfg=self.extra_generate_cfg,
                                 new_generate_cfg=extra_generate_cfg,
                             ))

    def _call_tool(self, tool_name: str, tool_args: Union[str, dict] = '{}', **kwargs) -> Union[str, List[ContentItem]]:
        """The interface of calling tools for the agent.

        Args:
            tool_name: The name of one tool.
            tool_args: Model generated or user given tool parameters.

        Returns:
            The output of tools.
        """
        with self._function_map_lock:
            if tool_name not in self.function_map:
                return f'Tool {tool_name} does not exists.'
            tool = self.function_map[tool_name]
        try:
            tool_result = tool.call(tool_args, **kwargs)
        except (ToolServiceError, DocParserError) as ex:
            raise ex
        except Exception as ex:
            exception_type = type(ex).__name__
            exception_message = str(ex)
            traceback_info = ''.join(traceback.format_tb(ex.__traceback__))
            error_message = f'An error occurred when calling tool `{tool_name}`:\n' \
                            f'{exception_type}: {exception_message}\n' \
                            f'Traceback:\n{traceback_info}'
            logger.warning(error_message)
            return error_message

        if isinstance(tool_result, str):
            return tool_result
        elif isinstance(tool_result, list) and all(isinstance(item, ContentItem) for item in tool_result):
            return tool_result  # multimodal tool results
        else:
            return json.dumps(tool_result, ensure_ascii=False, indent=4)

    def _init_tool(self, tool: Union[str, Dict, BaseTool]):
        if isinstance(tool, BaseTool):
            tool_name = tool.name
            with self._function_map_lock:
                if tool_name in self.function_map:
                    logger.warning(f'Repeatedly adding tool {tool_name}, will use the newest tool in function list')
                self.function_map[tool_name] = tool
        elif isinstance(tool, dict) and 'mcpServers' in tool:
            tools = MCPManager().initConfig(tool)
            with self._function_map_lock:
                for t in tools:
                    tool_name = t.name
                    if tool_name in self.function_map:
                        logger.warning(f'Repeatedly adding tool {tool_name}, will use the newest tool in function list')
                    self.function_map[tool_name] = t
        else:
            if isinstance(tool, dict):
                tool_name = tool['name']
                tool_cfg = tool
            else:
                tool_name = tool
                tool_cfg = None
            if tool_name not in TOOL_REGISTRY:
                raise ValueError(f'Tool {tool_name} is not registered.')

            tool_instance = TOOL_REGISTRY[tool_name](tool_cfg)
            with self._function_map_lock:
                if tool_name in self.function_map:
                    logger.warning(f'Repeatedly adding tool {tool_name}, will use the newest tool in function list')
                self.function_map[tool_name] = tool_instance

    def _detect_tool(self, message: Message) -> Tuple[bool, str, str, str]:
        """A built-in tool call detection for func_call format message.

        Args:
            message: one message generated by LLM.

        Returns:
            Need to call tool or not, tool name, tool args, text replies.
        """
        func_name = None
        func_args = None

        if message.function_call:
            func_call = message.function_call
            func_name = func_call.name
            func_args = func_call.arguments
        text = message.content
        if not text:
            text = ''

        return (func_name is not None), func_name, func_args, text

    # ==================== Hot-swappable Tool Management APIs ====================

    def add_tool(self, tool: Union[str, Dict, BaseTool], overwrite: bool = True) -> str:
        """Add a tool to the agent at runtime.

        This method allows hot-plugging tools while the agent is running.
        Thread-safe operation.

        Args:
            tool: Tool to add. Can be:
                - str: Tool name registered in TOOL_REGISTRY (e.g., 'code_interpreter')
                - dict: Tool configuration with 'name' key (e.g., {'name': 'code_interpreter', 'timeout': 10})
                - dict: MCP configuration with 'mcpServers' key
                - BaseTool: Direct tool instance
            overwrite: If True, overwrite existing tool with same name. If False, raise error.

        Returns:
            The name of the added tool (or comma-separated names for MCP tools).

        Raises:
            ValueError: If tool name already exists and overwrite=False.
            ValueError: If tool name is not in TOOL_REGISTRY.
        """
        if isinstance(tool, BaseTool):
            tool_name = tool.name
            with self._function_map_lock:
                if tool_name in self.function_map and not overwrite:
                    raise ValueError(f'Tool {tool_name} already exists. Set overwrite=True to replace it.')
                if tool_name in self.function_map:
                    logger.info(f'Replacing existing tool: {tool_name}')
                else:
                    logger.info(f'Adding new tool: {tool_name}')
                self.function_map[tool_name] = tool
            return tool_name
        elif isinstance(tool, dict) and 'mcpServers' in tool:
            tools = MCPManager().initConfig(tool)
            added_names = []
            with self._function_map_lock:
                for t in tools:
                    tool_name = t.name
                    if tool_name in self.function_map and not overwrite:
                        raise ValueError(f'Tool {tool_name} already exists. Set overwrite=True to replace it.')
                    if tool_name in self.function_map:
                        logger.info(f'Replacing existing tool: {tool_name}')
                    else:
                        logger.info(f'Adding new tool: {tool_name}')
                    self.function_map[tool_name] = t
                    added_names.append(tool_name)
            return ','.join(added_names)
        else:
            if isinstance(tool, dict):
                tool_name = tool['name']
                tool_cfg = tool
            else:
                tool_name = tool
                tool_cfg = None
            if tool_name not in TOOL_REGISTRY:
                raise ValueError(f'Tool {tool_name} is not registered in TOOL_REGISTRY.')

            tool_instance = TOOL_REGISTRY[tool_name](tool_cfg)
            with self._function_map_lock:
                if tool_name in self.function_map and not overwrite:
                    raise ValueError(f'Tool {tool_name} already exists. Set overwrite=True to replace it.')
                if tool_name in self.function_map:
                    logger.info(f'Replacing existing tool: {tool_name}')
                else:
                    logger.info(f'Adding new tool: {tool_name}')
                self.function_map[tool_name] = tool_instance
            return tool_name

    def remove_tool(self, tool_name: str) -> bool:
        """Remove a tool from the agent at runtime.

        This method allows hot-unplugging tools while the agent is running.
        Thread-safe operation.

        Args:
            tool_name: The name of the tool to remove.

        Returns:
            True if the tool was removed, False if it didn't exist.
        """
        with self._function_map_lock:
            if tool_name in self.function_map:
                del self.function_map[tool_name]
                logger.info(f'Removed tool: {tool_name}')
                return True
            else:
                logger.warning(f'Tool {tool_name} not found, nothing to remove.')
                return False

    def update_tool(self, tool: Union[str, Dict, BaseTool]) -> str:
        """Update/replace an existing tool at runtime.

        This is equivalent to add_tool with overwrite=True.
        Thread-safe operation.

        Args:
            tool: The new tool to replace the existing one. Same formats as add_tool.

        Returns:
            The name of the updated tool.

        Raises:
            ValueError: If the tool doesn't exist. Use add_tool for new tools.
        """
        # Determine the tool name first
        if isinstance(tool, BaseTool):
            tool_name = tool.name
        elif isinstance(tool, dict) and 'mcpServers' in tool:
            # For MCP, we just call add_tool with overwrite
            return self.add_tool(tool, overwrite=True)
        elif isinstance(tool, dict):
            tool_name = tool['name']
        else:
            tool_name = tool

        with self._function_map_lock:
            if tool_name not in self.function_map:
                raise ValueError(f'Tool {tool_name} does not exist. Use add_tool() to add new tools.')

        return self.add_tool(tool, overwrite=True)

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool by name.

        Thread-safe operation.

        Args:
            tool_name: The name of the tool.

        Returns:
            The tool instance if found, None otherwise.
        """
        with self._function_map_lock:
            return self.function_map.get(tool_name)

    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool exists.

        Thread-safe operation.

        Args:
            tool_name: The name of the tool.

        Returns:
            True if the tool exists, False otherwise.
        """
        with self._function_map_lock:
            return tool_name in self.function_map

    def list_tools(self) -> List[str]:
        """Get a list of all tool names.

        Thread-safe operation.

        Returns:
            A list of tool names currently registered with the agent.
        """
        with self._function_map_lock:
            return list(self.function_map.keys())

    def get_tools_info(self) -> List[Dict]:
        """Get detailed information about all tools.

        Thread-safe operation.

        Returns:
            A list of dictionaries containing tool information (name, description, parameters).
        """
        with self._function_map_lock:
            return [tool.function for tool in self.function_map.values()]

    def clear_tools(self) -> int:
        """Remove all tools from the agent.

        Thread-safe operation.

        Returns:
            The number of tools that were removed.
        """
        with self._function_map_lock:
            count = len(self.function_map)
            self.function_map.clear()
            logger.info(f'Cleared all {count} tools.')
            return count


# The most basic form of an agent is just a LLM, not augmented with any tool or workflow.
class BasicAgent(Agent):

    def _run(self, messages: List[Message], lang: str = 'en', **kwargs) -> Iterator[List[Message]]:
        extra_generate_cfg = {'lang': lang}
        if kwargs.get('seed') is not None:
            extra_generate_cfg['seed'] = kwargs['seed']
        return self._call_llm(messages, extra_generate_cfg=extra_generate_cfg)
