from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import medgemma_runtime
from medgemma_runtime import build_generate_text_fn, infer_runtime_config


def _torch_stub(*, available: bool, capability: tuple[int, int] = (9, 0)):
    cuda = SimpleNamespace(
        is_available=lambda: available,
        get_device_capability=lambda _index: capability,
        get_device_name=lambda _index: "Test GPU",
        device_count=lambda: 2,
    )
    return SimpleNamespace(cuda=cuda)


def test_runtime_config_uses_fp8_and_environment_tensor_parallelism() -> None:
    env = {
        "ECG_GEMMA_QUANTIZATION": "auto",
        "ECG_GEMMA_TP_SIZE": "2",
    }
    with (
        patch.dict(sys.modules, {"torch": _torch_stub(available=True)}),
        patch.dict(os.environ, env, clear=False),
    ):
        config = infer_runtime_config()

    assert config == {
        "device_name": "Test GPU",
        "device_count": 2,
        "compute_capability": "9.0",
        "quantization": "fp8",
        "tensor_parallel_size": 2,
    }


def test_runtime_config_allows_quantization_to_be_disabled() -> None:
    env = {
        "ECG_GEMMA_QUANTIZATION": "none",
        "ECG_GEMMA_TP_SIZE": "",
    }
    with (
        patch.dict(
            sys.modules,
            {"torch": _torch_stub(available=True, capability=(8, 0))},
        ),
        patch.dict(os.environ, env, clear=False),
    ):
        config = infer_runtime_config()

    assert config["quantization"] is None
    assert config["tensor_parallel_size"] == 1


def test_runtime_config_rejects_cpu_only_environment() -> None:
    with patch.dict(sys.modules, {"torch": _torch_stub(available=False)}):
        with pytest.raises(RuntimeError, match="No CUDA device"):
            infer_runtime_config()


def test_runtime_config_rejects_invalid_tensor_parallel_size() -> None:
    with (
        patch.dict(sys.modules, {"torch": _torch_stub(available=True)}),
        patch.dict(
            os.environ,
            {"ECG_GEMMA_TP_SIZE": "0", "ECG_GEMMA_QUANTIZATION": "none"},
            clear=False,
        ),
    ):
        with pytest.raises(RuntimeError, match="positive integer"):
            infer_runtime_config()


def test_build_generate_text_fn_reuses_initialized_vllm() -> None:
    init_calls = []
    generate_calls = []

    class FakeLLM:
        def __init__(self, **kwargs) -> None:
            init_calls.append(kwargs)

        def generate(self, prompts, sampling_params):
            generate_calls.append((prompts, sampling_params))
            output = SimpleNamespace(text="generated diagnosis")
            return [SimpleNamespace(outputs=[output])]

    class FakeSamplingParams:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    vllm_stub = SimpleNamespace(
        LLM=FakeLLM,
        SamplingParams=FakeSamplingParams,
    )
    runtime_config = {
        "device_name": "Test GPU",
        "device_count": 1,
        "compute_capability": "9.0",
        "quantization": "fp8",
        "tensor_parallel_size": 2,
    }
    with (
        patch.dict(sys.modules, {"vllm": vllm_stub}),
        patch.object(
            medgemma_runtime,
            "infer_runtime_config",
            return_value=runtime_config,
        ),
    ):
        generate = build_generate_text_fn(
            "test-model",
            model_max_len=4096,
            max_output_tokens=128,
        )
        first = generate("first prompt")
        second = generate("second prompt")

    assert first == "generated diagnosis"
    assert second == "generated diagnosis"
    assert init_calls == [
        {
            "model": "test-model",
            "dtype": "bfloat16",
            "max_model_len": 4096,
            "tensor_parallel_size": 2,
            "quantization": "fp8",
        }
    ]
    assert [call[0] for call in generate_calls] == [
        ["first prompt"],
        ["second prompt"],
    ]
    assert generate_calls[0][1].kwargs == {
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 128,
    }
