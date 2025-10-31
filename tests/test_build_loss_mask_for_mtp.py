import importlib.util
import pathlib
import sys
import types

import pytest
import torch

import slime.backends  # Ensure parent package is registered.

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_MEGATRON_TRAINING_PATH = _REPO_ROOT / "slime" / "backends" / "megatron_utils"


def _ensure_stub_modules() -> None:
    """Provide minimal stubs for optional Megatron modules."""
    # Stub megatron.core.mpu to satisfy imports that are unused in tests.
    mpu_module = types.ModuleType("megatron.core.mpu")
    mpu_module._cp_rank = 0
    mpu_module._cp_world_size = 1

    def _get_cp_world_size(*_, **__):
        return mpu_module._cp_world_size

    def _get_cp_rank(*_, **__):
        return mpu_module._cp_rank

    mpu_module.get_context_parallel_rank = _get_cp_rank
    mpu_module.get_context_parallel_world_size = _get_cp_world_size
    mpu_module.get_data_parallel_world_size = lambda *_, **__: 1
    mpu_module.get_data_parallel_group = lambda *_, **__: None
    mpu_module.get_data_parallel_rank = lambda *_, **__: 0
    mpu_module.get_tensor_model_parallel_rank = lambda *_, **__: 0
    mpu_module.get_pipeline_model_parallel_rank = lambda *_, **__: 0
    mpu_module.get_pipeline_model_parallel_world_size = lambda *_, **__: 1
    mpu_module.is_pipeline_last_stage = lambda *_, **__: True
    sys.modules.setdefault("megatron.core.mpu", mpu_module)
    try:
        import megatron.core as core_module  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        core_module = types.ModuleType("megatron.core")
        sys.modules.setdefault("megatron.core", core_module)
    else:
        core_module = sys.modules["megatron.core"]
    setattr(core_module, "mpu", mpu_module)

    # Stub megatron.training package hierarchy.
    training_pkg = types.ModuleType("megatron.training")
    training_pkg.__path__ = []
    sys.modules.setdefault("megatron.training", training_pkg)

    checkpoint_module = types.ModuleType("megatron.training.checkpointing")
    checkpoint_module.load_checkpoint = lambda *_, **__: None
    checkpoint_module.save_checkpoint = lambda *_, **__: None
    sys.modules.setdefault("megatron.training.checkpointing", checkpoint_module)

    global_vars_module = types.ModuleType("megatron.training.global_vars")
    global_vars_module.get_args = lambda: types.SimpleNamespace()
    sys.modules.setdefault("megatron.training.global_vars", global_vars_module)

    training_module = types.ModuleType("megatron.training.training")
    training_module.get_model = lambda *_, **__: []
    sys.modules.setdefault("megatron.training.training", training_module)

    # Register package placeholder to avoid executing heavy __init__.
    package_name = "slime.backends.megatron_utils"
    if package_name not in sys.modules:
        package_module = types.ModuleType(package_name)
        package_module.__path__ = [str(_MEGATRON_TRAINING_PATH)]
        sys.modules[package_name] = package_module
        setattr(slime.backends, "megatron_utils", package_module)

    # Provide lightweight stubs for modules only needed for import-time wiring.
    checkpoint_stub = types.ModuleType(f"{package_name}.checkpoint")
    checkpoint_stub.load_checkpoint = lambda *_, **__: None
    checkpoint_stub.save_checkpoint = lambda *_, **__: None
    sys.modules.setdefault(f"{package_name}.checkpoint", checkpoint_stub)

    cp_utils_name = f"{package_name}.cp_utils"
    if cp_utils_name not in sys.modules:
        cp_utils_spec = importlib.util.spec_from_file_location(cp_utils_name, _MEGATRON_TRAINING_PATH / "cp_utils.py")
        cp_utils_module = importlib.util.module_from_spec(cp_utils_spec)
        sys.modules[cp_utils_name] = cp_utils_module
        assert cp_utils_spec.loader is not None
        cp_utils_spec.loader.exec_module(cp_utils_module)
    else:
        cp_utils_module = sys.modules[cp_utils_name]
    setattr(sys.modules[package_name], "cp_utils", cp_utils_module)

    data_stub = types.ModuleType(f"{package_name}.data")

    class _DummyIterator:  # pragma: no cover - helper only
        pass

    data_stub.DataIterator = _DummyIterator
    data_stub.get_batch = lambda *_, **__: None
    sys.modules.setdefault(f"{package_name}.data", data_stub)

    loss_stub = types.ModuleType(f"{package_name}.loss")
    loss_stub.loss_function = lambda *_, **__: None
    sys.modules.setdefault(f"{package_name}.loss", loss_stub)

    model_provider_stub = types.ModuleType(f"{package_name}.model_provider")
    model_provider_stub.get_model_provider_func = lambda *_, **__: None
    sys.modules.setdefault(f"{package_name}.model_provider", model_provider_stub)


_ensure_stub_modules()

_MODEL_SPEC = importlib.util.spec_from_file_location(
    "slime.backends.megatron_utils.model", _MEGATRON_TRAINING_PATH / "model.py"
)
model_module = importlib.util.module_from_spec(_MODEL_SPEC)
sys.modules[_MODEL_SPEC.name] = model_module
assert _MODEL_SPEC.loader is not None
_MODEL_SPEC.loader.exec_module(model_module)


def _extract_build_loss_mask_for_mtp() -> types.FunctionType:
    forward_step_code = next(
        const
        for const in model_module.train_one_step.__code__.co_consts
        if isinstance(const, types.CodeType) and const.co_name == "forward_step"
    )
    build_loss_mask_code = next(
        const
        for const in forward_step_code.co_consts
        if isinstance(const, types.CodeType) and const.co_name == "build_loss_mask_for_mtp"
    )
    assert not build_loss_mask_code.co_freevars, "build_loss_mask_for_mtp unexpectedly closed over variables."
    return types.FunctionType(build_loss_mask_code, model_module.__dict__)


@pytest.fixture(name="build_loss_mask_for_mtp")
def fixture_build_loss_mask_for_mtp() -> types.FunctionType:
    return _extract_build_loss_mask_for_mtp()


@pytest.fixture(autouse=True)
def reset_context_parallel_state():
    mpu_module = sys.modules["megatron.core.mpu"]
    prev_world_size = mpu_module._cp_world_size
    prev_rank = mpu_module._cp_rank
    mpu_module._cp_world_size = 1
    mpu_module._cp_rank = 0
    try:
        yield
    finally:
        mpu_module._cp_world_size = prev_world_size
        mpu_module._cp_rank = prev_rank


@pytest.fixture
def slice_with_cp_calls(monkeypatch):
    calls: list[tuple[torch.Tensor, float, torch.Tensor]] = []

    original_slice_with_cp = model_module.slice_with_cp

    def wrapped_slice_with_cp(tensor: torch.Tensor, pad_value: float) -> torch.Tensor:
        result = original_slice_with_cp(tensor, pad_value)
        calls.append((tensor.clone(), pad_value, result.clone()))
        return result

    monkeypatch.setattr(model_module, "slice_with_cp", wrapped_slice_with_cp)
    return calls


def test_build_loss_mask_for_mtp_single_sequence(build_loss_mask_for_mtp, slice_with_cp_calls):
    tokens = torch.zeros(1, 8, dtype=torch.float32)
    batch = {
        "tokens": tokens,
        "total_lengths": [6],
        "response_lengths": [2],
        "loss_masks": [torch.tensor([1.0, 0.5], dtype=torch.float32)],
    }

    loss_mask = build_loss_mask_for_mtp(batch)

    expected = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0]], dtype=torch.float32)
    assert torch.allclose(loss_mask, expected)
    assert loss_mask.shape == tokens.shape

    assert len(slice_with_cp_calls) == 1
    input_mask, pad_value, output_mask = slice_with_cp_calls[0]
    assert pad_value == 0.0
    assert torch.allclose(input_mask, torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.5], dtype=torch.float32))
    assert torch.allclose(output_mask, input_mask)


def test_build_loss_mask_for_mtp_multiple_sequences(build_loss_mask_for_mtp, slice_with_cp_calls):
    tokens = torch.zeros(1, 10, dtype=torch.float32)
    batch = {
        "tokens": tokens,
        "total_lengths": [5, 4],
        "response_lengths": [2, 1],
        "loss_masks": [
            torch.tensor([0.2, 1.0], dtype=torch.float32),
            torch.tensor([0.3], dtype=torch.float32),
        ],
    }

    loss_mask = build_loss_mask_for_mtp(batch)

    expected = torch.tensor([[0.0, 0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0, 0.3, 0.0]], dtype=torch.float32)
    assert torch.allclose(loss_mask, expected)
    assert loss_mask.shape == tokens.shape

    assert len(slice_with_cp_calls) == 2
    first_call_input, _, first_call_output = slice_with_cp_calls[0]
    second_call_input, _, second_call_output = slice_with_cp_calls[1]
    expected_first = torch.tensor([0.0, 0.0, 0.0, 0.2, 1.0], dtype=torch.float32)
    expected_second = torch.tensor([0.0, 0.0, 0.0, 0.3], dtype=torch.float32)
    assert torch.allclose(first_call_input, expected_first)
    assert torch.allclose(first_call_output, expected_first)
    assert torch.allclose(second_call_input, expected_second)
    assert torch.allclose(second_call_output, expected_second)


def test_build_loss_mask_for_mtp_invalid_loss_mask_size(build_loss_mask_for_mtp, slice_with_cp_calls):
    tokens = torch.zeros(1, 5, dtype=torch.float32)
    batch = {
        "tokens": tokens,
        "total_lengths": [5],
        "response_lengths": [3],
        "loss_masks": [torch.tensor([1.0, 0.5], dtype=torch.float32)],
    }

    with pytest.raises(AssertionError, match="Unexpected loss mask size"):
        build_loss_mask_for_mtp(batch)

    assert not slice_with_cp_calls


def test_build_loss_mask_for_mtp_with_context_parallel_slice(
    build_loss_mask_for_mtp, slice_with_cp_calls, monkeypatch
):
    mpu_module = sys.modules["megatron.core.mpu"]
    monkeypatch.setattr(mpu_module, "_cp_world_size", 2, raising=False)
    monkeypatch.setattr(mpu_module, "_cp_rank", 1, raising=False)

    tokens = torch.zeros(1, 16, dtype=torch.float32)
    response_mask = torch.tensor([1.0, 1.0, 0.5, 0.2], dtype=torch.float32)
    batch = {
        "tokens": tokens,
        "total_lengths": [9],
        "response_lengths": [4],
        "loss_masks": [response_mask],
    }

    loss_mask = build_loss_mask_for_mtp(batch)

    expected_chunk = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.5, 0.2], dtype=torch.float32)
    assert torch.allclose(loss_mask[:, :6], expected_chunk)
    assert torch.allclose(loss_mask[:, 6:], torch.zeros_like(loss_mask[:, 6:]))

    assert len(slice_with_cp_calls) == 1
    input_mask, pad_value, output_mask = slice_with_cp_calls[0]
    assert pad_value == 0.0
    assert torch.allclose(
        input_mask,
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.5, 0.2], dtype=torch.float32),
    )
    assert torch.allclose(output_mask, expected_chunk)
