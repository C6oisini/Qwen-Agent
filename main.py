import os
import logging
import warnings

from dotenv import load_dotenv

# 禁用所有警告
warnings.filterwarnings('ignore')
from qwen_agent.agents import (
    Assistant, ReActChat, GroupChat, Router,
    Orchestrator, AgentScheduler, ScheduledTask, TaskTemplates
)
from qwen_agent.gui import WebUI
from qwen_agent.agent import Agent as BaseAgent
from qwen_agent.tools import MCPManager
from qwen_agent.log import logger as qwen_logger

# 注册自定义工具
import tools.rag_pipeline_tool  # noqa: F401

load_dotenv()

# 关闭所有不必要的终端输出
# 调试模式：设置为 True 可以看到 Orchestrator 的调试信息
DEBUG_ORCHESTRATOR = True
if DEBUG_ORCHESTRATOR:
    qwen_logger.setLevel(logging.INFO)
else:
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

tf_tools = filter_tools(all_mcp_tools, keywords=['ad-create_tt_ad', 'ad-rag_search', 'ad-query_top_materials', 'ad-get_available_indicators', 'ad-get_app_info'])
fx_tools = filter_tools(all_mcp_tools, keywords=['ad-query_ad_data', 'ad-change_creative_status', 'ad-get_available_indicators', 'ad-get_app_info'])
pj_tools = filter_tools(all_mcp_tools, keywords=['ad-query_ad_data', 'ad-update_tt_projects', 'ad-get_available_indicators', 'ad-get_app_info'])

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

with open('instruction/project-instruction.md', 'r', encoding='utf-8') as f:
    project_instruction = f.read()
project_agent = Assistant(
    llm=llm_cfg,
    name='头条项目管理助手',
    description='你是一个头条广告项目管理助手，帮助用户进行项目分析、投放计划调整。',
    system_message=project_instruction,
    function_list=pj_tools,  # 使用已初始化的工具列表
)

general_agent = Assistant(
    llm=llm_cfg,
    name='通用助手',
    description='处理一般性问题、知识问答、文本写作、翻译等不需要特殊工具的任务.'
)

# =============== 选择运行模式 ===============
# 模式1: Router - 简单路由，选择一个 agent 处理（原来的方式）
# 模式2: Orchestrator - 协调模式，支持多 agent 交互
# 模式3: Orchestrator + Scheduler - 协调 + 定时任务

USE_ORCHESTRATOR = True  # 设置为 True 使用协调模式
USE_SCHEDULER = False     # 设置为 True 启用定时任务

# 所有专业 agent 列表
all_agents = [tf_agent, fx_agent, rag_agent, project_agent, general_agent]

if USE_ORCHESTRATOR:
    # 使用协调器模式 - 支持 Agent 间交互
    active_bot = Orchestrator(
        llm=llm_cfg,
        agents=all_agents,
        name='广告系统协调者',
        description='协调多个专家 Agent 完成广告投放、分析、优化等任务'
    )
    print("🤖 已启用 Orchestrator 协调模式 - 支持多 Agent 交互")
else:
    # 使用简单路由模式（原来的方式）
    active_bot = Router(
        llm=llm_cfg,
        agents=all_agents,
    )
    print("🤖 已启用 Router 路由模式")

# # 定时任务调度器
# scheduler = None
# if USE_SCHEDULER:
#     # 创建 agent 名称到实例的映射
#     agents_map = {agent.name: agent for agent in all_agents}

#     # 初始化调度器
#     scheduler = AgentScheduler(
#         agents=agents_map,
#         storage_path='./z-scheduler',
#         on_task_complete=lambda ex: print(f"✅ 任务完成: {ex.task_id} - {ex.status.value}")
#     )

#     # 添加预定义的定时任务
#     # 1. 每小时数据分析
#     try:
#         scheduler.add_task(TaskTemplates.hourly_analysis("头条广告分析助手"))
#     except Exception as e:
#         print(f"任务已存在或添加失败: {e}")

#     # 2. 每日报告
#     try:
#         scheduler.add_task(TaskTemplates.daily_report("头条广告分析助手"))
#     except Exception as e:
#         print(f"任务已存在或添加失败: {e}")

#     # 3. 素材自动入库（每30分钟）
#     try:
#         scheduler.add_task(TaskTemplates.material_sync("RAG素材入库助手"))
#     except Exception as e:
#         print(f"任务已存在或添加失败: {e}")

#     # 启动调度器
#     scheduler.start()
#     print("⏰ 已启用定时任务调度器")
#     print(f"   当前任务: {list(scheduler.tasks.keys())}")

chatbot_config = {
    'user.name': 'User',
    'input.placeholder': '请输入您的问题...',
    'history.max_length': 20,   
    'history.session_dir': './z-sessions',   
    'verbose': False,
}

if __name__ == '__main__':
    WebUI(active_bot, chatbot_config=chatbot_config).run(share=True)
