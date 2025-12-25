import pprint
import urllib.parse
import json5
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.utils.output_beautify import typewriter_print
from dotenv import load_dotenv
load_dotenv()


llm_cfg = {
    # 使用 DashScope 提供的模型服务：
    'model': 'qwen-max-latest',
    'model_type': 'qwen_dashscope',
    'api_key': os.environ.get('DASHSCOPE_API_KEY'),
    'generate_cfg': {
        'top_p': 0.8
    }
}
with open('instruction.md', 'r', encoding='utf-8') as f:
    system_instruction = f.read()
mcp_config = {
    "mcpServers": {
        "ad": {
            "url": "http://localhost:8000/mcp",
            "type": "streamable-http"
        },
    }
}
tools = ['code_interpreter', mcp_config]  # `code_interpreter` 是框架自带的工具，用于执行代码。
bot = Assistant(llm=llm_cfg,
                system_message=system_instruction,
                function_list=tools,
                )


# 启动 WebUI 界面
from qwen_agent.gui import WebUI
WebUI(bot).run(share=True)