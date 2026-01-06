import os
import logging
import warnings

from dotenv import load_dotenv

# 禁用所有警告
warnings.filterwarnings('ignore')
from qwen_agent.agents import Assistant, ReActChat, GroupChat, Router
from qwen_agent.gui import WebUI
from qwen_agent.agent import Agent as BaseAgent
from qwen_agent.tools import MCPManager
from qwen_agent.log import logger as qwen_logger

# 注册自定义工具
import tools.rag_pipeline_tool  # noqa: F401

load_dotenv()

# 关闭所有不必要的终端输出
qwen_logger.setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.ERROR)
logging.getLogger('openai').setLevel(logging.ERROR)
logging.getLogger('httpcore').setLevel(logging.ERROR)
logging.getLogger('gradio').setLevel(logging.ERROR)
_original_call_llm = BaseAgent._call_llm
def _call_llm_nonstream(self, messages, functions=None, stream=True, extra_generate_cfg=None):
    responses = _original_call_llm(self,
                                   messages=messages,
                                   functions=functions,
                                   stream=False,
                                   extra_generate_cfg=extra_generate_cfg)
     
    if isinstance(responses, list):
        return iter([responses])
    return responses
BaseAgent._call_llm = _call_llm_nonstream

llm_cfg = {
    'model': 'qwen/qwen3-max',
    'model_server': 'https://openrouter.ai/api/v1',
    'api_key': os.environ.get('OR_API_KEY'),
    'generate_cfg': {
        'top_p': 0.8
    }
}


mcp_config = {
    "mcpServers": {
        "ad": {
            "url": "http://localhost:8000/mcp",
            "type": "streamable-http"
        },
    }
}

# 分发工具
all_mcp_tools = MCPManager().initConfig(mcp_config)
print(f"已加载 MCP 工具: {[t.name for t in all_mcp_tools]}")


def filter_tools(tools, keywords=None, exclude_keywords=None):
    result = tools
    if keywords:
        result = [t for t in result if any(kw in t.name for kw in keywords)]
    if exclude_keywords:
        result = [t for t in result if not any(kw in t.name for kw in exclude_keywords)]
    return result

tf_tools = filter_tools(all_mcp_tools, keywords=['ad-create_tt_ad', 'ad-rag_search', 'ad-query_top_materials', 'ad-get_available_indicators'])
fx_tools = filter_tools(all_mcp_tools, keywords=['ad-query_ad_data', 'ad-change_creative_status', 'ad-get_available_indicators'])


# 定义agent
with open('instruction/tf-instruction.md', 'r', encoding='utf-8') as f:
    tf_instruction = f.read()
tf_agent = Assistant(
    llm=llm_cfg,
    name='头条广告投放助手',
    description='你是专门负责分析用户请求，帮助用户完成投放广告需求的助手。你可以使用外部的工具和MCP服务来获取数据和执行任务。',
    system_message=tf_instruction,
    function_list=tf_tools,  # 使用已初始化的工具列表
)

with open('instruction/fx-instruction.md', 'r', encoding='utf-8') as f:
    fx_instruction = f.read()
fx_agent = Assistant(
    llm=llm_cfg,
    name='头条广告分析助手',
    description='你是专门负责分析广告投放数据、生成报告和执行复杂任务的助手。适合处理需要深入分析和综合信息的任务。',
    system_message=fx_instruction,
    function_list=fx_tools,  # 使用已初始化的工具列表
)

with open('instruction/rag-instruction.md', 'r', encoding='utf-8') as f:
    rag_instruction = f.read()
rag_agent = Assistant(
    llm=llm_cfg,
    name='RAG素材入库助手',
    description='你是一个检测新增素材，并对其进行caption和embedding入库的智能助手。',
    system_message=rag_instruction,
    function_list=['rag_pipeline'],  # 使用自定义的 pipeline 工具
)


general_agent = Assistant(
    llm=llm_cfg,
    name='通用助手',
    description='处理一般性问题、知识问答、文本写作、翻译等不需要特殊工具的任务.'
)

router_bot = Router(
    llm=llm_cfg,
    agents=[tf_agent, fx_agent, rag_agent, general_agent],
)


active_bot = router_bot

chatbot_config = {
    'user.name': 'User',
    'input.placeholder': '请输入您的问题...',
    'history.max_length': 20,   
    'history.session_dir': './z-sessions',   
    'verbose': False,
}

if __name__ == '__main__':
    WebUI(active_bot, chatbot_config=chatbot_config).run(share=True)
