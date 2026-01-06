"""
RAG Pipeline 工具：将 pipeline.py 的功能包装成 Qwen-Agent 工具
"""

import json
from typing import Union

from qwen_agent.tools.base import BaseTool, register_tool


@register_tool('rag_pipeline')
class RagPipelineTool(BaseTool):
    """素材入库 Pipeline 工具"""

    name = 'rag_pipeline'
    description = '增量同步素材到向量数据库。从 analytics.material_dim 抽取新素材，生成 caption 和 embedding 后存入 material.embeddings 表。'
    parameters = {
        'type': 'object',
        'properties': {
            'limit': {
                'type': 'integer',
                'description': '最多处理的素材数量，不填则处理所有缺失素材'
            }
        },
        'required': []
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        """执行素材入库 pipeline"""
        params_json = self._verify_json_format_args(params)
        limit = params_json.get('limit', None)

        # 导入 pipeline 模块并执行
        from tools.pipeline import run_pipeline

        try:
            summary = run_pipeline(limit=limit)
            return json.dumps(summary, ensure_ascii=False, indent=2)
        except Exception as e:
            return json.dumps({
                'success': False,
                'error': str(e)
            }, ensure_ascii=False)
