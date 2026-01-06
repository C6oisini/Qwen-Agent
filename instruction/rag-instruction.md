# RAG 素材同步 Pipeline 助手

增量同步素材 embedding 的完整 pipeline，支持从素材库抽取数据、生成 caption、计算 embedding 并入库。

## 功能概述

1. **素材抽取**：从 `analytics.material_dim` 抽取满足 `tt_valid=true` 且 `media_type=2` 的视频素材
2. **增量同步**：仅处理 `material.embeddings` 中不存在的素材
3. **Caption 生成**：使用 Qwen3-VL 生成封面描述、视频描述和广告文案
4. **Embedding 计算**：使用 NVIDIA NVClip 生成多模态 embedding
5. **入库存储**：将处理结果写入 `material.embeddings` 表



## 使用方法

### 运行 Pipeline

```bash
# 处理所有缺失的素材
cd ~/Downloads/tt-mcp-only/.claude/skills/rag-material-pipeline
uv run python scripts/pipeline.py
```

### 在代码中调用

```python
from scripts.pipeline import run_pipeline

# 运行完整 pipeline
summary = run_pipeline()
print(f"处理完成: {summary['inserted']} 条成功, {len(summary['errors'])} 条失败")

# 限制处理数量
summary = run_pipeline(limit=50)
```

## 数据流程

```
analytics.material_dim (源表)
    │
    ▼ 筛选: tt_valid=true AND media_type=2
    │
    ▼ 过滤: 不在 material.embeddings 中的记录
    │
    ├─► cover_url ─► Qwen3-VL ─► cover_caption
    │
    ├─► file_url ─► Qwen3-VL ─► file_caption
    │
    ├─► cover_caption + file_caption ─► Qwen3-VL ─► ad_caption
    │
    ├─► 各 caption ─► NVClip ─► text embeddings
    │
    ├─► cover_url ─► NVClip ─► image embedding
    │
    ▼ 加权融合 (各 25%)
    │
    ▼ 写入 material.embeddings
```

## 输出结构

每条素材处理后生成以下字段：

| 字段 | 说明 |
|------|------|
| `gdt_material_id` | 广点通素材 ID |
| `tt_material_id` | 头条素材 ID |
| `cover_caption` | 封面图片描述 |
| `file_caption` | 视频内容描述 |
| `ad_caption` | 生成的广告文案 |
| `embedding` | 融合后的 768 维向量 |
| `embedding_model` | nvidia/nvclip |

## 返回结果

`run_pipeline()` 返回一个 summary 字典：

```python
{
    "requested_limit": 10,      # 请求处理的数量限制
    "total_candidates": 10,     # 实际找到的待处理素材数
    "inserted": 8,              # 成功插入的数量
    "errors": [                 # 错误列表
        {"gdt_material_id": "xxx", "error": "错误信息"}
    ]
}
```

## 依赖服务

- **PostgreSQL**：素材元数据和 embedding 存储
- **ModelScope Qwen3-VL**：图片/视频描述生成
- **NVIDIA NVClip API**：多模态 embedding 计算

## 注意事项

1. **API 限流**：ModelScope 和 NVIDIA API 有调用频率限制，大批量处理时注意控制速度
2. **网络超时**：视频描述生成可能耗时较长，默认超时 60 秒
3. **错误处理**：单条素材处理失败不会中断整个 pipeline，错误会记录在 summary 中
4. **幂等性**：使用 `ON CONFLICT DO NOTHING`，重复运行不会产生重复数据
