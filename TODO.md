# TODO

## 1:当Orchestrator call 0 个Agent的时候程序会卡死
[Orchestrator] Plan content preview: 感谢您的提醒！我注意到之前查询的是2024-12-29的数据，但实际上今天是2026-01-09。让我重新查询今天（2026-01-09）的创意投放数据。

...
[Orchestrator] Parsed 0 agent calls
[Orchestrator] No agent calls found in plan, returning

## 2:Router Agent有时候不知道选哪个Agent
弄清Router的交互逻辑，类似dfs还是？具不具备回溯的能力，每次用户新的提问到的是RouterAgent还是上一轮对话的Agent
补充提示词 or 改架构

## 3:记忆模块和Prompt存储
supabase构建日志记录和向量数据库根据用户数据快速找到合适的预定义prompt