import inspect

import pytest

from predictive_cache.storage import ExpertLocation, ExpertStore


def test_expert_store_is_abstract():
    assert inspect.isabstract(ExpertStore)


def test_expert_store_declares_required_operations():
    required = {
        "load",
        "unload",
        "prefetch",
        "location",
    }

    assert required.issubset(set(dir(ExpertStore)))


@pytest.mark.parametrize(
    "method_name",
    ["load", "unload", "prefetch", "location"],
)
def test_expert_store_operations_are_abstract(method_name):
    method = getattr(ExpertStore, method_name)

    assert getattr(method, "__isabstractmethod__", False) is True


def test_expert_location_values():
    assert ExpertLocation.GPU.value == "gpu"
    assert ExpertLocation.RAM.value == "ram"
    assert ExpertLocation.NVME.value == "nvme"
    assert ExpertLocation.REMOTE.value == "remote"


def test_expert_location_is_string_compatible():
    assert isinstance(ExpertLocation.GPU, str)
    assert ExpertLocation.GPU == "gpu"

def test_expert_store_defines_record_crud_contract():
    assert hasattr(ExpertStore, "put")
    assert hasattr(ExpertStore, "get")
    assert hasattr(ExpertStore, "contains")
    assert hasattr(ExpertStore, "remove")