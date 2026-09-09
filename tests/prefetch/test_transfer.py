from predictive_cache.prefetch import (
    PrefetchSource,
    PrefetchTarget,
    PrefetchTask,
    PrefetchTransferExecutor,
)


def test_transfer_routes_nvme_to_ram():
    calls: list[int] = []

    def default_handler(task: PrefetchTask) -> None:
        calls.append(-1)

    def nvme_to_ram(task: PrefetchTask) -> None:
        calls.append(task.expert_id)

    transfer = PrefetchTransferExecutor(default_handler)

    transfer.register(
        PrefetchSource.NVME,
        PrefetchTarget.RAM,
        nvme_to_ram,
    )

    task = PrefetchTask(
        expert_id=1,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.RAM,
    )

    transfer.execute(task)

    assert calls == [1]


def test_transfer_routes_ram_to_gpu():
    calls: list[int] = []

    def default_handler(task: PrefetchTask) -> None:
        calls.append(-1)

    def ram_to_gpu(task: PrefetchTask) -> None:
        calls.append(task.expert_id)

    transfer = PrefetchTransferExecutor(default_handler)

    transfer.register(
        PrefetchSource.RAM,
        PrefetchTarget.GPU,
        ram_to_gpu,
    )

    task = PrefetchTask(
        expert_id=2,
        source=PrefetchSource.RAM,
        target=PrefetchTarget.GPU,
    )

    transfer.execute(task)

    assert calls == [2]


def test_transfer_routes_nvme_to_gpu():
    calls: list[int] = []

    def default_handler(task: PrefetchTask) -> None:
        calls.append(-1)

    def nvme_to_gpu(task: PrefetchTask) -> None:
        calls.append(task.expert_id)

    transfer = PrefetchTransferExecutor(default_handler)

    transfer.register(
        PrefetchSource.NVME,
        PrefetchTarget.GPU,
        nvme_to_gpu,
    )

    task = PrefetchTask(
        expert_id=3,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.GPU,
    )

    transfer.execute(task)

    assert calls == [3]

def test_transfer_accepts_callable_handler():
    calls: list[int] = []

    class NvmeToRamHandler:
        def __call__(self, task: PrefetchTask) -> None:
            calls.append(task.expert_id)

    transfer = PrefetchTransferExecutor(
        lambda task: None,
    )

    transfer.register(
        PrefetchSource.NVME,
        PrefetchTarget.RAM,
        NvmeToRamHandler(),
    )

    task = PrefetchTask(
        expert_id=10,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.RAM,
    )

    transfer.execute(task)

    assert calls == [10]

def test_transfer_falls_back_for_unregistered_route():
    calls: list[str] = []

    def default_handler(task: PrefetchTask) -> None:
        calls.append("default")

    def nvme_to_ram(task: PrefetchTask) -> None:
        calls.append("nvme_to_ram")

    transfer = PrefetchTransferExecutor(default_handler)

    transfer.register(
        PrefetchSource.NVME,
        PrefetchTarget.RAM,
        nvme_to_ram,
    )

    task = PrefetchTask(
        expert_id=20,
        source=PrefetchSource.RAM,
        target=PrefetchTarget.GPU,
    )

    transfer.execute(task)

    assert calls == ["default"]

def test_transfer_routes_all_planned_paths():
    calls: list[tuple[PrefetchSource, PrefetchTarget, int]] = []

    def default_handler(task: PrefetchTask) -> None:
        raise AssertionError("default handler should not be used")

    transfer = PrefetchTransferExecutor(default_handler)

    def make_handler(
        source: PrefetchSource,
        target: PrefetchTarget,
    ):
        def handler(task: PrefetchTask) -> None:
            calls.append(
                (
                    source,
                    target,
                    task.expert_id,
                )
            )

        return handler

    transfer.register(
        PrefetchSource.NVME,
        PrefetchTarget.RAM,
        make_handler(
            PrefetchSource.NVME,
            PrefetchTarget.RAM,
        ),
    )

    transfer.register(
        PrefetchSource.RAM,
        PrefetchTarget.GPU,
        make_handler(
            PrefetchSource.RAM,
            PrefetchTarget.GPU,
        ),
    )

    transfer.register(
        PrefetchSource.NVME,
        PrefetchTarget.GPU,
        make_handler(
            PrefetchSource.NVME,
            PrefetchTarget.GPU,
        ),
    )

    tasks = [
        PrefetchTask(
            expert_id=1,
            source=PrefetchSource.NVME,
            target=PrefetchTarget.RAM,
        ),
        PrefetchTask(
            expert_id=2,
            source=PrefetchSource.RAM,
            target=PrefetchTarget.GPU,
        ),
        PrefetchTask(
            expert_id=3,
            source=PrefetchSource.NVME,
            target=PrefetchTarget.GPU,
        ),
    ]

    for task in tasks:
        transfer.execute(task)

    assert calls == [
        (
            PrefetchSource.NVME,
            PrefetchTarget.RAM,
            1,
        ),
        (
            PrefetchSource.RAM,
            PrefetchTarget.GPU,
            2,
        ),
        (
            PrefetchSource.NVME,
            PrefetchTarget.GPU,
            3,
        ),
    ]