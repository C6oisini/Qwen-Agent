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
定时任务调度器 - 支持长期运行的周期性 Agent 任务

功能:
1. 定时执行 Agent 任务（如每小时分析一次数据）
2. 支持任务链：分析 → 决策 → 执行
3. 持久化任务状态和历史
4. 支持任务回调和通知
"""

import json
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any
from enum import Enum
import logging

from qwen_agent import Agent
from qwen_agent.llm.schema import Message

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ScheduledTask:
    """定时任务配置"""
    task_id: str
    name: str
    description: str
    agent_name: str  # 执行任务的 Agent 名称
    prompt: str  # 给 Agent 的指令
    interval_seconds: int  # 执行间隔（秒）
    enabled: bool = True
    last_run: Optional[str] = None
    next_run: Optional[str] = None
    run_count: int = 0
    # 任务链：执行完当前任务后触发的下一个任务
    chain_tasks: List[str] = field(default_factory=list)
    # 条件执行：只有满足条件才执行后续任务
    chain_condition: Optional[str] = None  # e.g., "contains:优化建议"
    # 回调
    on_complete: Optional[str] = None  # callback name
    metadata: Dict = field(default_factory=dict)


@dataclass
class TaskExecution:
    """任务执行记录"""
    execution_id: str
    task_id: str
    start_time: str
    end_time: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    triggered_tasks: List[str] = field(default_factory=list)


class AgentScheduler:
    """Agent 任务调度器"""

    def __init__(self,
                 agents: Dict[str, Agent],
                 storage_path: str = "./z-scheduler",
                 on_task_complete: Optional[Callable[[TaskExecution], None]] = None):
        """
        Args:
            agents: Agent 名称到实例的映射
            storage_path: 任务状态持久化路径
            on_task_complete: 任务完成回调
        """
        self.agents = agents
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.tasks: Dict[str, ScheduledTask] = {}
        self.execution_history: List[TaskExecution] = []
        self.on_task_complete = on_task_complete

        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # 加载持久化的任务
        self._load_tasks()

    def add_task(self, task: ScheduledTask) -> str:
        """添加定时任务"""
        with self._lock:
            if task.agent_name not in self.agents:
                raise ValueError(f"Agent not found: {task.agent_name}")

            task.next_run = datetime.now().isoformat()
            self.tasks[task.task_id] = task
            self._save_tasks()
            logger.info(f"Task added: {task.task_id} - {task.name}")
            return task.task_id

    def remove_task(self, task_id: str) -> bool:
        """移除定时任务"""
        with self._lock:
            if task_id in self.tasks:
                del self.tasks[task_id]
                self._save_tasks()
                logger.info(f"Task removed: {task_id}")
                return True
            return False

    def enable_task(self, task_id: str, enabled: bool = True):
        """启用/禁用任务"""
        with self._lock:
            if task_id in self.tasks:
                self.tasks[task_id].enabled = enabled
                self._save_tasks()

    def start(self):
        """启动调度器"""
        if self._running:
            logger.warning("Scheduler already running")
            return

        self._running = True
        self._scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self._scheduler_thread.start()
        logger.info("Scheduler started")

    def stop(self):
        """停止调度器"""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        logger.info("Scheduler stopped")

    def run_task_now(self, task_id: str) -> Optional[TaskExecution]:
        """立即执行任务"""
        if task_id not in self.tasks:
            logger.error(f"Task not found: {task_id}")
            return None

        task = self.tasks[task_id]
        return self._execute_task(task)

    def _run_scheduler(self):
        """调度器主循环"""
        while self._running:
            try:
                now = datetime.now()
                tasks_to_run = []

                with self._lock:
                    for task in self.tasks.values():
                        if not task.enabled:
                            continue
                        if task.next_run and datetime.fromisoformat(task.next_run) <= now:
                            tasks_to_run.append(task)

                for task in tasks_to_run:
                    self._execute_task(task)

                # 每 10 秒检查一次
                time.sleep(10)

            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(30)

    def _execute_task(self, task: ScheduledTask) -> TaskExecution:
        """执行单个任务"""
        execution = TaskExecution(
            execution_id=f"{task.task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            task_id=task.task_id,
            start_time=datetime.now().isoformat(),
            status=TaskStatus.RUNNING
        )

        logger.info(f"Executing task: {task.name}")

        try:
            agent = self.agents.get(task.agent_name)
            if not agent:
                raise ValueError(f"Agent not found: {task.agent_name}")

            # 构建消息
            messages = [{"role": "user", "content": task.prompt}]

            # 添加上下文（如果有上次执行的结果）
            if task.last_run and task.metadata.get('last_result'):
                context = f"\n\n【上次执行结果】({task.last_run}):\n{task.metadata['last_result'][:500]}..."
                messages[0]['content'] += context

            # 调用 Agent
            response = agent.run_nonstream(messages)

            result = ""
            if response:
                result = response[-1].get('content', '') if isinstance(response[-1], dict) else response[-1].content

            execution.status = TaskStatus.SUCCESS
            execution.result = result
            execution.end_time = datetime.now().isoformat()

            # 更新任务状态
            with self._lock:
                task.last_run = datetime.now().isoformat()
                task.next_run = (datetime.now() +
                                 __import__('datetime').timedelta(seconds=task.interval_seconds)).isoformat()
                task.run_count += 1
                task.metadata['last_result'] = result[:1000]  # 保存最近结果的摘要
                self._save_tasks()

            # 检查是否需要触发链式任务
            if task.chain_tasks and self._check_chain_condition(task, result):
                for chain_task_id in task.chain_tasks:
                    if chain_task_id in self.tasks:
                        logger.info(f"Triggering chain task: {chain_task_id}")
                        chain_execution = self._execute_task(self.tasks[chain_task_id])
                        execution.triggered_tasks.append(chain_task_id)

        except Exception as e:
            execution.status = TaskStatus.FAILED
            execution.error = str(e)
            execution.end_time = datetime.now().isoformat()
            logger.error(f"Task execution failed: {task.name} - {e}")

        # 保存执行历史
        self.execution_history.append(execution)
        self._save_execution_history()

        # 触发回调
        if self.on_task_complete:
            try:
                self.on_task_complete(execution)
            except Exception as e:
                logger.error(f"Callback error: {e}")

        return execution

    def _check_chain_condition(self, task: ScheduledTask, result: str) -> bool:
        """检查是否满足链式任务触发条件"""
        if not task.chain_condition:
            return True

        condition = task.chain_condition
        if condition.startswith("contains:"):
            keyword = condition.replace("contains:", "")
            return keyword in result
        elif condition.startswith("not_contains:"):
            keyword = condition.replace("not_contains:", "")
            return keyword not in result

        return True

    def _save_tasks(self):
        """持久化任务配置"""
        tasks_file = self.storage_path / "tasks.json"
        tasks_data = {tid: asdict(t) for tid, t in self.tasks.items()}
        with open(tasks_file, 'w', encoding='utf-8') as f:
            json.dump(tasks_data, f, ensure_ascii=False, indent=2)

    def _load_tasks(self):
        """加载持久化的任务"""
        tasks_file = self.storage_path / "tasks.json"
        if tasks_file.exists():
            try:
                with open(tasks_file, 'r', encoding='utf-8') as f:
                    tasks_data = json.load(f)
                for tid, data in tasks_data.items():
                    self.tasks[tid] = ScheduledTask(**data)
                logger.info(f"Loaded {len(self.tasks)} tasks")
            except Exception as e:
                logger.error(f"Failed to load tasks: {e}")

    def _save_execution_history(self):
        """保存执行历史（只保留最近 100 条）"""
        history_file = self.storage_path / "execution_history.json"
        recent = self.execution_history[-100:]
        history_data = []
        for ex in recent:
            ex_dict = asdict(ex)
            ex_dict['status'] = ex.status.value
            history_data.append(ex_dict)
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)

    def get_status(self) -> Dict:
        """获取调度器状态"""
        return {
            "running": self._running,
            "total_tasks": len(self.tasks),
            "enabled_tasks": sum(1 for t in self.tasks.values() if t.enabled),
            "total_executions": len(self.execution_history),
            "tasks": {
                tid: {
                    "name": t.name,
                    "enabled": t.enabled,
                    "last_run": t.last_run,
                    "next_run": t.next_run,
                    "run_count": t.run_count
                }
                for tid, t in self.tasks.items()
            }
        }


# 预定义的常用任务模板
class TaskTemplates:
    """常用任务模板"""

    @staticmethod
    def hourly_analysis(agent_name: str = "头条广告分析助手") -> ScheduledTask:
        """每小时数据分析"""
        return ScheduledTask(
            task_id="hourly_analysis",
            name="每小时数据分析",
            description="每小时分析广告投放数据，发现异常和优化机会",
            agent_name=agent_name,
            prompt="""请分析最近1小时的广告投放数据:
1. 查询各项目的消耗、转化、ROI数据
2. 对比历史同期数据，识别异常波动
3. 找出表现不佳的广告计划
4. 给出具体的优化建议

请按以下格式输出:
【数据概览】...
【异常告警】...
【优化建议】...
""",
            interval_seconds=3600,  # 1小时
            chain_tasks=["auto_optimize"],
            chain_condition="contains:优化建议"
        )

    @staticmethod
    def auto_optimize(agent_name: str = "头条项目管理助手") -> ScheduledTask:
        """自动优化任务"""
        return ScheduledTask(
            task_id="auto_optimize",
            name="自动优化执行",
            description="根据分析结果自动执行优化操作",
            agent_name=agent_name,
            prompt="""根据最近的分析结果，执行以下优化操作:
1. 对表现差的计划降低预算或暂停
2. 对表现好的计划提高预算
3. 记录所有执行的操作

注意：只执行高置信度的优化建议，不确定的操作请跳过。
""",
            interval_seconds=86400,  # 不自动执行，由链式触发
            enabled=False  # 默认不启用自动执行
        )

    @staticmethod
    def daily_report(agent_name: str = "头条广告分析助手") -> ScheduledTask:
        """每日报告"""
        return ScheduledTask(
            task_id="daily_report",
            name="每日数据报告",
            description="生成每日投放数据汇总报告",
            agent_name=agent_name,
            prompt="""请生成今日广告投放数据报告:
1. 汇总今日总消耗、总转化、整体ROI
2. 按项目分析各项目表现
3. 与昨日/上周同期对比
4. 本周趋势分析
5. 下一步投放建议

请生成结构化的报告，便于快速了解投放情况。
""",
            interval_seconds=86400,  # 24小时
        )

    @staticmethod
    def material_sync(agent_name: str = "RAG素材入库助手") -> ScheduledTask:
        """素材同步任务"""
        return ScheduledTask(
            task_id="material_sync",
            name="素材自动入库",
            description="定期检查新素材并入库",
            agent_name=agent_name,
            prompt="请检查是否有新增素材需要处理，如果有，执行 caption 和 embedding 入库流程。",
            interval_seconds=1800,  # 30分钟
        )
