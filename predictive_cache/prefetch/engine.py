from __future__ import annotations

import threading
from typing import Callable

from .queue import PrefetchQueue
from .transfer import PrefetchTransferExecutor
from .types import (
    PrefetchResult,
    PrefetchStatus,
    PrefetchTask,
)


Loader = Callable[[PrefetchTask], None]


class PrefetchEngine:
    """
    后台 Expert Prefetch Engine。

    生命周期：

        start()
            ↓
        RUNNING
            ↓
        stop()
            ↓
        STOPPING
            ↓
        Queue Drain
            ↓
        STOPPED

    4D-4:
    - Task State Tracking

    4D-5:
    - QUEUED -> CANCELLED
    - 取消尚未开始执行的任务
    - RUNNING 任务不强制杀线程
    """

    def __init__(
            self,
            loader: Loader,
            *,
            num_workers: int = 1,
            max_queue_size: int | None = None,
            transfer_executor: PrefetchTransferExecutor | None = None,
    ) -> None:
        if num_workers <= 0:
            raise ValueError("num_workers must be > 0")

        if max_queue_size is not None and max_queue_size <= 0:
            raise ValueError("max_queue_size must be > 0")

        self._loader = loader
        self._transfer_executor = (
            transfer_executor
            if transfer_executor is not None
            else PrefetchTransferExecutor(loader)
        )
        self._num_workers = num_workers

        self._queue = PrefetchQueue(
            max_size=max_queue_size,
        )

        self._workers: list[threading.Thread] = []
        self._stop_event = threading.Event()
        self._running = False

        self._lock = threading.Lock()
        self._results: list[PrefetchResult] = []
        self._task_states: dict[int, PrefetchStatus] = {}

    # ---------------------------------------------------------
    # lifecycle
    # ---------------------------------------------------------

    def start(self) -> None:
        """
        启动 Prefetch Worker。

        已经运行时保持幂等。
        """

        with self._lock:
            if self._running:
                return

            self._stop_event.clear()

            self._workers = []

            for index in range(self._num_workers):
                worker = threading.Thread(
                    target=self._worker_loop,
                    name=f"prefetch-worker-{index}",
                    daemon=True,
                )

                worker.start()

                self._workers.append(worker)

            self._running = True

    def stop(self, *, wait: bool = True) -> None:
        """
        Graceful shutdown。

        已经提交的任务继续执行。

        新任务在 stop() 后会被拒绝。
        """

        with self._lock:
            if not self._running:
                return

            self._stop_event.set()

            workers = list(self._workers)

        if wait:
            self._queue.join()

            self._queue.notify_all()

            for worker in workers:
                worker.join()

            with self._lock:
                self._workers.clear()
                self._running = False

        else:
            self._queue.notify_all()

    # ---------------------------------------------------------
    # submit
    # ---------------------------------------------------------

    def submit(self, task: PrefetchTask) -> bool:
        """
        提交 PrefetchTask。

        成功：

            task -> QUEUED

        失败：

            状态保持不变。
        """

        with self._lock:
            if self._stop_event.is_set():
                return False

            if not self._running:
                return False

            accepted = self._queue.put(task)

            if not accepted:
                return False

            # 必须在持有 _lock 时设置状态。
            #
            # Worker 如果已经取到 task，
            # 会等待同一把 lock。
            #
            # 这样可以避免：
            #
            # QUEUED -> RUNNING
            #
            # 被 submit() 又覆盖回 QUEUED。
            self._task_states[task.expert_id] = (
                PrefetchStatus.QUEUED
            )

            return True

    # ---------------------------------------------------------
    # cancellation
    # ---------------------------------------------------------

    def cancel(self, expert_id: int) -> bool:
        """
        取消一个尚未开始执行的 Task。

        QUEUED:
            -> CANCELLED

        RUNNING:
            -> 不取消

        COMPLETED:
            -> 不取消

        FAILED:
            -> 不取消

        UNKNOWN:
            -> False
        """

        with self._lock:
            status = self._task_states.get(expert_id)

            if status != PrefetchStatus.QUEUED:
                return False

            cancelled = self._queue.cancel(expert_id)

            if not cancelled:
                return False

            self._task_states[expert_id] = (
                PrefetchStatus.CANCELLED
            )

            return True

    # ---------------------------------------------------------
    # worker
    # ---------------------------------------------------------

    def _worker_loop(self) -> None:
        """
        Worker 主循环。

        STOPPING 状态下继续 Drain Queue。

        只有：

            stop requested
            +
            no unfinished tasks

        才退出。
        """

        while True:
            task = self._queue.get(timeout=0.2)

            if task is None:
                if (
                    self._stop_event.is_set()
                    and self._queue.unfinished_tasks == 0
                ):
                    break

                continue

            # -------------------------------------------------
            # QUEUED -> RUNNING
            # -------------------------------------------------

            self._set_task_state(
                task.expert_id,
                PrefetchStatus.RUNNING,
            )

            result: PrefetchResult | None = None

            try:
                result = self._execute(task)

                with self._lock:
                    self._results.append(result)

            finally:
                # _execute() 正常情况下永远返回 result。
                #
                # 如果未来 _execute() 自身发生未预期异常，
                # 仍然必须 task_done()，避免 join() 永久阻塞。
                if result is not None:
                    self._set_task_state(
                        task.expert_id,
                        result.status,
                    )

                self._queue.task_done()

    # ---------------------------------------------------------
    # execution
    # ---------------------------------------------------------

    def _execute(
        self,
        task: PrefetchTask,
    ) -> PrefetchResult:
        """
        执行一个 PrefetchTask。

        loader 异常转换为 FAILED。
        """

        try:
            self._transfer_executor.execute(task)

            return PrefetchResult(
                expert_id=task.expert_id,
                status=PrefetchStatus.COMPLETED,
                source=task.source,
                target=task.target,
                priority=task.priority,
            )

        except Exception as exc:
            return PrefetchResult(
                expert_id=task.expert_id,
                status=PrefetchStatus.FAILED,
                source=task.source,
                target=task.target,
                priority=task.priority,
                error=str(exc),
            )

    # ---------------------------------------------------------
    # state tracking
    # ---------------------------------------------------------

    def _set_task_state(
        self,
        expert_id: int,
        status: PrefetchStatus,
    ) -> None:
        with self._lock:
            self._task_states[expert_id] = status

    def task_status(
        self,
        expert_id: int,
    ) -> PrefetchStatus | None:
        """
        获取指定 Task 状态。
        """

        with self._lock:
            return self._task_states.get(expert_id)

    def task_states(
        self,
    ) -> dict[int, PrefetchStatus]:
        """
        返回全部 Task State 的副本。
        """

        with self._lock:
            return dict(self._task_states)

    @property
    def queued_tasks(self) -> list[int]:
        with self._lock:
            return [
                expert_id
                for expert_id, status
                in self._task_states.items()
                if status == PrefetchStatus.QUEUED
            ]

    @property
    def running_tasks(self) -> list[int]:
        with self._lock:
            return [
                expert_id
                for expert_id, status
                in self._task_states.items()
                if status == PrefetchStatus.RUNNING
            ]

    @property
    def completed_tasks(self) -> list[int]:
        with self._lock:
            return [
                expert_id
                for expert_id, status
                in self._task_states.items()
                if status == PrefetchStatus.COMPLETED
            ]

    @property
    def failed_tasks(self) -> list[int]:
        with self._lock:
            return [
                expert_id
                for expert_id, status
                in self._task_states.items()
                if status == PrefetchStatus.FAILED
            ]

    @property
    def cancelled_tasks(self) -> list[int]:
        """
        返回已经取消的 expert_id。
        """

        with self._lock:
            return [
                expert_id
                for expert_id, status
                in self._task_states.items()
                if status == PrefetchStatus.CANCELLED
            ]

    def clear_task_states(self) -> None:
        """
        清空 Task State Tracking。

        不影响 queue 和 results。
        """

        with self._lock:
            self._task_states.clear()

    # inspection
    @property
    def max_queue_size(self) -> int | None:
        return self._queue.max_size

    @property
    def queue_full(self) -> bool:
        return self._queue.is_full

    @property
    def queue_size(self) -> int:
        return self._queue.size()

    @property
    def unfinished_tasks(self) -> int:
        return self._queue.unfinished_tasks

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    def results(self) -> list[PrefetchResult]:
        with self._lock:
            return list(self._results)

    def clear_results(self) -> None:
        with self._lock:
            self._results.clear()