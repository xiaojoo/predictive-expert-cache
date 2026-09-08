from __future__ import annotations

import heapq
import threading
from typing import Optional

from .types import PrefetchTask


class PrefetchQueue:
    """
    Thread-safe priority queue.

    priority 越高，越优先执行。

    4D-3:
    - unfinished task tracking
    - task_done()
    - join()
    - graceful shutdown / queue drain

    4D-5:
    - 支持取消尚未被 Worker 取出的任务
    """

    def __init__(
            self,
            max_size: int | None = None,
    ) -> None:
        if max_size is not None and max_size <= 0:
            raise ValueError("max_size must be > 0")

        self._max_size = max_size

        self._heap: list[tuple[float, int, PrefetchTask]] = []

        self._sequence = 0

        self._condition = threading.Condition()

        self._pending: set[int] = set()

        # 已经 put 但尚未 task_done() 的任务数量。
        #
        # 包括：
        # - QUEUED
        # - RUNNING
        self._unfinished_tasks = 0

    # ---------------------------------------------------------
    # put / get
    # ---------------------------------------------------------

    def put(self, task: PrefetchTask) -> bool:
        """
        添加任务。

        返回：
            True  -> 成功加入
            False -> expert 已经 pending
            或 queue 已满
        """

        with self._condition:
            if task.expert_id in self._pending:
                return False

            if (
                self._max_size is not None
                and len(self._heap) >= self._max_size
            ):
                return False

            heapq.heappush(
                self._heap,
                (
                    -task.priority,
                    self._sequence,
                    task,
                ),
            )

            self._sequence += 1

            self._pending.add(task.expert_id)

            self._unfinished_tasks += 1

            self._condition.notify()

            return True

    def get(
        self,
        timeout: Optional[float] = None,
    ) -> Optional[PrefetchTask]:
        """
        获取最高优先级任务。

        get() 不会减少 unfinished_tasks。

        Worker 执行完成后必须调用 task_done()。
        """

        with self._condition:
            if not self._heap:
                self._condition.wait(timeout)

            if not self._heap:
                return None

            _, _, task = heapq.heappop(self._heap)

            self._pending.discard(task.expert_id)

            return task

    # ---------------------------------------------------------
    # task tracking
    # ---------------------------------------------------------

    def task_done(self) -> None:
        """
        标记一个已经 get() 的任务完成。

        每一次成功 get() 最终必须对应一次 task_done()。
        """

        with self._condition:
            if self._unfinished_tasks <= 0:
                raise ValueError(
                    "task_done() called too many times"
                )

            self._unfinished_tasks -= 1

            if self._unfinished_tasks == 0:
                self._condition.notify_all()

    def join(self) -> None:
        """
        阻塞直到所有任务完成或被取消。
        """

        with self._condition:
            while self._unfinished_tasks > 0:
                self._condition.wait()

    # ---------------------------------------------------------
    # cancellation
    # ---------------------------------------------------------

    def cancel(self, expert_id: int) -> bool:
        """
        取消一个尚未被 Worker 取出的任务。

        返回：
            True:
                成功取消

            False:
                任务不存在于 pending queue 中，
                通常意味着：
                - 不存在
                - 已经被 Worker 取走
                - 已经取消

        注意：

        这里会真正从 heap 中删除任务，
        并同时减少 unfinished_tasks。

        因此被 cancel() 成功的任务：
            不需要再调用 task_done()。
        """

        with self._condition:
            if expert_id not in self._pending:
                return False

            # 从 heap 中找到目标任务。
            index = None

            for position, (_, _, task) in enumerate(
                self._heap
            ):
                if task.expert_id == expert_id:
                    index = position
                    break

            if index is None:
                # 理论上不应该发生。
                #
                # _pending 和 _heap 应该保持一致。
                self._pending.discard(expert_id)
                return False

            # 删除 heap 中的任务。
            self._heap[index] = self._heap[-1]
            self._heap.pop()

            if index < len(self._heap):
                heapq.heapify(self._heap)

            self._pending.discard(expert_id)

            # 被取消的任务不再需要 Worker 执行，
            # 所以直接完成 unfinished tracking。
            if self._unfinished_tasks <= 0:
                raise ValueError(
                    "unfinished task count is invalid"
                )

            self._unfinished_tasks -= 1

            if self._unfinished_tasks == 0:
                self._condition.notify_all()

            return True

    # ---------------------------------------------------------
    # inspection
    # ---------------------------------------------------------

    def contains(self, expert_id: int) -> bool:
        with self._condition:
            return expert_id in self._pending

    def size(self) -> int:
        with self._condition:
            return len(self._heap)

    @property
    def unfinished_tasks(self) -> int:
        with self._condition:
            return self._unfinished_tasks

    @property
    def max_size(self) -> int | None:
        return self._max_size

    @property
    def is_full(self) -> bool:
        with self._condition:
            if self._max_size is None:
                return False

            return len(self._heap) >= self._max_size

    def clear(self) -> None:
        """
        清空队列。

        当前设计中 clear() 将所有尚未开始执行的任务
        视为被丢弃。

        注意：
        如果存在正在执行的任务，不应该调用 clear()。
        """

        with self._condition:
            self._heap.clear()
            self._pending.clear()

            self._unfinished_tasks = 0

            self._condition.notify_all()

    def notify_all(self) -> None:
        """
        唤醒所有正在等待 queue 的 Worker。
        """

        with self._condition:
            self._condition.notify_all()