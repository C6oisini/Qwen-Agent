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

"""Tests for hot-swappable tool management APIs."""

import threading
import time

import pytest

from qwen_agent.agent import Agent, BasicAgent
from qwen_agent.tools.base import BaseTool


class DummyTool(BaseTool):
    """A simple tool for testing purposes."""
    name = 'dummy_tool'
    description = 'A dummy tool that returns a fixed message.'
    parameters = {
        'type': 'object',
        'properties': {
            'message': {
                'type': 'string',
                'description': 'The message to echo'
            }
        },
        'required': ['message']
    }

    def call(self, params, **kwargs):
        import json
        if isinstance(params, str):
            params = json.loads(params)
        return f"Dummy: {params.get('message', 'no message')}"


class AnotherTool(BaseTool):
    """Another tool for testing tool updates."""
    name = 'another_tool'
    description = 'Another tool for testing.'
    parameters = {
        'type': 'object',
        'properties': {
            'value': {
                'type': 'integer',
                'description': 'A numeric value'
            }
        },
        'required': ['value']
    }

    def call(self, params, **kwargs):
        import json
        if isinstance(params, str):
            params = json.loads(params)
        return f"Value: {params.get('value', 0)}"


class UpdatedDummyTool(BaseTool):
    """Updated version of DummyTool."""
    name = 'dummy_tool'  # Same name as DummyTool
    description = 'An updated dummy tool with new behavior.'
    parameters = {
        'type': 'object',
        'properties': {
            'message': {
                'type': 'string',
                'description': 'The message to process'
            }
        },
        'required': ['message']
    }

    def call(self, params, **kwargs):
        import json
        if isinstance(params, str):
            params = json.loads(params)
        return f"Updated Dummy: {params.get('message', 'no message').upper()}"


class TestToolHotSwap:
    """Test suite for tool hot-swap functionality."""

    def setup_method(self):
        """Setup a basic agent for each test."""
        self.agent = BasicAgent()

    def test_add_tool_with_instance(self):
        """Test adding a tool using a BaseTool instance."""
        assert len(self.agent.list_tools()) == 0

        tool = DummyTool()
        name = self.agent.add_tool(tool)

        assert name == 'dummy_tool'
        assert self.agent.has_tool('dummy_tool')
        assert len(self.agent.list_tools()) == 1
        assert 'dummy_tool' in self.agent.list_tools()

    def test_add_tool_with_string(self):
        """Test adding a tool using a registered tool name."""
        # 'image_gen' is a built-in tool registered in TOOL_REGISTRY
        name = self.agent.add_tool('image_gen')
        assert name == 'image_gen'
        assert self.agent.has_tool('image_gen')

    def test_add_tool_with_dict_config(self):
        """Test adding a tool using a dictionary configuration."""
        name = self.agent.add_tool({'name': 'image_gen'})
        assert name == 'image_gen'
        assert self.agent.has_tool('image_gen')

    def test_add_tool_overwrite_default(self):
        """Test that adding a duplicate tool overwrites by default."""
        self.agent.add_tool(DummyTool())
        original_tool = self.agent.get_tool('dummy_tool')

        updated_tool = UpdatedDummyTool()
        self.agent.add_tool(updated_tool)

        current_tool = self.agent.get_tool('dummy_tool')
        assert current_tool is updated_tool
        assert current_tool is not original_tool

    def test_add_tool_no_overwrite_raises(self):
        """Test that adding a duplicate tool with overwrite=False raises error."""
        self.agent.add_tool(DummyTool())

        with pytest.raises(ValueError, match='already exists'):
            self.agent.add_tool(UpdatedDummyTool(), overwrite=False)

    def test_add_tool_unregistered_raises(self):
        """Test that adding an unregistered tool name raises error."""
        with pytest.raises(ValueError, match='not registered'):
            self.agent.add_tool('nonexistent_tool')

    def test_remove_tool_existing(self):
        """Test removing an existing tool."""
        self.agent.add_tool(DummyTool())
        assert self.agent.has_tool('dummy_tool')

        result = self.agent.remove_tool('dummy_tool')
        assert result is True
        assert not self.agent.has_tool('dummy_tool')
        assert len(self.agent.list_tools()) == 0

    def test_remove_tool_nonexistent(self):
        """Test removing a non-existent tool returns False."""
        result = self.agent.remove_tool('nonexistent')
        assert result is False

    def test_update_tool(self):
        """Test updating an existing tool."""
        self.agent.add_tool(DummyTool())
        original = self.agent.get_tool('dummy_tool')

        updated = UpdatedDummyTool()
        self.agent.update_tool(updated)

        current = self.agent.get_tool('dummy_tool')
        assert current is updated
        assert current is not original

    def test_update_tool_nonexistent_raises(self):
        """Test that updating a non-existent tool raises error."""
        with pytest.raises(ValueError, match='does not exist'):
            self.agent.update_tool(DummyTool())

    def test_get_tool(self):
        """Test getting a tool by name."""
        tool = DummyTool()
        self.agent.add_tool(tool)

        retrieved = self.agent.get_tool('dummy_tool')
        assert retrieved is tool

    def test_get_tool_nonexistent(self):
        """Test getting a non-existent tool returns None."""
        result = self.agent.get_tool('nonexistent')
        assert result is None

    def test_has_tool(self):
        """Test checking tool existence."""
        assert not self.agent.has_tool('dummy_tool')
        self.agent.add_tool(DummyTool())
        assert self.agent.has_tool('dummy_tool')

    def test_list_tools(self):
        """Test listing all tool names."""
        assert self.agent.list_tools() == []

        self.agent.add_tool(DummyTool())
        self.agent.add_tool(AnotherTool())

        tools = self.agent.list_tools()
        assert len(tools) == 2
        assert 'dummy_tool' in tools
        assert 'another_tool' in tools

    def test_get_tools_info(self):
        """Test getting detailed tool information."""
        self.agent.add_tool(DummyTool())

        info = self.agent.get_tools_info()
        assert len(info) == 1
        assert info[0]['name'] == 'dummy_tool'
        assert 'description' in info[0]
        assert 'parameters' in info[0]

    def test_clear_tools(self):
        """Test clearing all tools."""
        self.agent.add_tool(DummyTool())
        self.agent.add_tool(AnotherTool())
        assert len(self.agent.list_tools()) == 2

        count = self.agent.clear_tools()
        assert count == 2
        assert len(self.agent.list_tools()) == 0

    def test_call_tool_after_add(self):
        """Test that a dynamically added tool can be called."""
        self.agent.add_tool(DummyTool())

        result = self.agent._call_tool('dummy_tool', '{"message": "hello"}')
        assert result == 'Dummy: hello'

    def test_call_tool_after_remove(self):
        """Test that a removed tool cannot be called."""
        self.agent.add_tool(DummyTool())
        self.agent.remove_tool('dummy_tool')

        result = self.agent._call_tool('dummy_tool', '{}')
        assert 'does not exists' in result

    def test_call_tool_after_update(self):
        """Test that an updated tool has new behavior."""
        self.agent.add_tool(DummyTool())
        result1 = self.agent._call_tool('dummy_tool', '{"message": "hello"}')
        assert result1 == 'Dummy: hello'

        self.agent.update_tool(UpdatedDummyTool())
        result2 = self.agent._call_tool('dummy_tool', '{"message": "hello"}')
        assert result2 == 'Updated Dummy: HELLO'


class TestToolHotSwapThreadSafety:
    """Test thread safety of tool hot-swap operations."""

    def test_concurrent_add_remove(self):
        """Test concurrent add and remove operations."""
        agent = BasicAgent()
        errors = []
        iterations = 100

        def add_tools():
            try:
                for i in range(iterations):
                    tool = DummyTool()
                    tool.name = f'tool_{i % 10}'
                    agent.add_tool(tool, overwrite=True)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def remove_tools():
            try:
                for i in range(iterations):
                    agent.remove_tool(f'tool_{i % 10}')
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def list_tools():
            try:
                for _ in range(iterations):
                    _ = agent.list_tools()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_tools),
            threading.Thread(target=remove_tools),
            threading.Thread(target=list_tools),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"

    def test_concurrent_call_and_update(self):
        """Test calling a tool while it's being updated."""
        agent = BasicAgent()
        agent.add_tool(DummyTool())
        errors = []
        iterations = 50

        def call_tool():
            try:
                for _ in range(iterations):
                    result = agent._call_tool('dummy_tool', '{"message": "test"}')
                    # Result should be from either old or new tool, not corrupted
                    assert 'Dummy' in result or 'Updated' in result or 'does not exists' in result
                    time.sleep(0.002)
            except Exception as e:
                errors.append(e)

        def update_tool():
            try:
                for i in range(iterations):
                    if i % 2 == 0:
                        agent.add_tool(DummyTool(), overwrite=True)
                    else:
                        agent.add_tool(UpdatedDummyTool(), overwrite=True)
                    time.sleep(0.002)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=call_tool),
            threading.Thread(target=update_tool),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
