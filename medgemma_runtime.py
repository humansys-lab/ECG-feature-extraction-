from __future__ import annotations

import os
from typing import Any, Callable

from medgemma_ecg_core import DEFAULT_MAX_OUTPUT_TOKENS, DEFAULT_MODEL_MAX_LEN


def _tensor_parallel_size(raw_value: str) -> int:
    if not raw_value:
        return 1
    try:
        size = int(raw_value)
    except ValueError as exc:
        raise RuntimeError("ECG_GEMMA_TP_SIZE must be a positive integer") from exc
    if size < 1:
        raise RuntimeError("ECG_GEMMA_TP_SIZE must be a positive integer")
    return size


def infer_runtime_config() -> dict[str, Any]:
    """Resolve the shared CUDA, quantization, and tensor-parallel settings."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required to inspect the MedGemma GPU runtime."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA device was detected. `torch.cuda.is_available()` is False, "
            "so vLLM cannot start."
        )

    major, minor = torch.cuda.get_device_capability(0)
    quantization_env = os.getenv("ECG_GEMMA_QUANTIZATION", "").strip().lower()
    if quantization_env in {"", "auto"}:
        quantization = "fp8" if (major > 8 or (major == 8 and minor >= 9)) else None
    elif quantization_env in {"none", "no", "off"}:
        quantization = None
    else:
        quantization = quantization_env

    return {
        "device_name": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(),
        "compute_capability": f"{major}.{minor}",
        "quantization": quantization,
        "tensor_parallel_size": _tensor_parallel_size(
            os.getenv("ECG_GEMMA_TP_SIZE", "").strip()
        ),
    }


def build_generate_text_fn(
    model_path: str,
    *,
    model_max_len: int = DEFAULT_MODEL_MAX_LEN,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> Callable[[str], str]:
    """Initialize vLLM and return the text generator shared by CLI entrypoints."""
    try:
        from vllm import LLM, SamplingParams
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "vllm is required to run MedGemma diagnostics."
        ) from exc

    runtime_config = infer_runtime_config()
    llm_kwargs: dict[str, Any] = {
        "model": model_path,
        "dtype": "bfloat16",
        "max_model_len": model_max_len,
        "tensor_parallel_size": runtime_config["tensor_parallel_size"],
    }
    if runtime_config["quantization"]:
        llm_kwargs["quantization"] = runtime_config["quantization"]

    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        temperature=0.2,
        top_p=0.9,
        max_tokens=max_output_tokens,
    )

    def _generate(prompt: str) -> str:
        outputs = llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text

    return _generate
