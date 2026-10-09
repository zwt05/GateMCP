"""Local Transformers inference and offline diagnostic backends."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Generation:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    response_model: str | None = None


class ModelBackend(Protocol):
    name: str

    def generate(self, prompt: str) -> Generation: ...


@dataclass
class ScriptedBackend:
    responses: list[str]
    name: str = "scripted"
    _index: int = 0

    def generate(self, prompt: str) -> Generation:
        response = self.responses[self._index] if self._index < len(self.responses) else ""
        self._index += 1
        return Generation(response)


@dataclass
class StaticBackend:
    """Deterministic invalid backend used only for transport smoke tests."""

    response: str = ""
    name: str = "offline-static"
    max_new_tokens: int = 600

    def generate(self, prompt: str) -> Generation:
        return Generation(self.response)


@dataclass
class TransformersBackend:
    model_path: str
    temperature: float = 0.0
    max_new_tokens: int = 512
    enable_thinking: bool = False
    name: str = field(init=False)
    _tokenizer: object = field(init=False, repr=False)
    _model: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        path = Path(self.model_path).expanduser().resolve()
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"Local model is incomplete or missing: {path}")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install GPU requirements before local inference") from exc
        self.name = f"local-transformers:{path}"
        self._tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        checkpoint = json.loads((path / "config.json").read_text(encoding="utf-8"))
        quant_method = str(checkpoint.get("quantization_config", {}).get("quant_method", "")).lower()
        if torch.cuda.is_available():
            dtype = torch.float16 if quant_method == "awq" else (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
        else:
            dtype = torch.float32
        load_kwargs: dict[str, object] = {
            "local_files_only": True,
            "torch_dtype": dtype,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }
        # ``device_map=auto`` may fill nearly all VRAM with weights and then
        # fail when an offloaded output layer or KV cache is materialized for
        # generation.  Large-model workers can reserve generation headroom
        # without changing their decoding budget through this environment
        # variable.  CUDA_VISIBLE_DEVICES makes the selected worker GPU index 0.
        gpu_limit = os.environ.get("CODE_AB_GPU_MAX_MEMORY_GIB")
        if torch.cuda.is_available() and gpu_limit:
            limit = int(gpu_limit)
            if limit <= 0:
                raise ValueError("CODE_AB_GPU_MAX_MEMORY_GIB must be positive")
            load_kwargs["max_memory"] = {0: f"{limit}GiB", "cpu": "128GiB"}
        self._model = AutoModelForCausalLM.from_pretrained(path, **load_kwargs)
        self._model.eval()
        if self.temperature <= 0:
            # Some checkpoints ship sampling defaults even though greedy
            # decoding is requested.  Clearing ignored fields removes noisy
            # warnings without changing the deterministic token sequence.
            self._model.generation_config.temperature = None
            self._model.generation_config.top_p = None
            self._model.generation_config.top_k = None

    def generate(self, prompt: str) -> Generation:
        import torch

        messages = [{"role": "user", "content": prompt}]
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            rendered = self._tokenizer.apply_chat_template(
                messages, enable_thinking=self.enable_thinking, **kwargs
            )
        except TypeError:
            rendered = self._tokenizer.apply_chat_template(messages, **kwargs)
        encoded = self._tokenizer([rendered], return_tensors="pt")
        device = self._model.get_input_embeddings().weight.device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        generation_kwargs: dict[str, object] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature
        with torch.inference_mode():
            output = self._model.generate(**encoded, **generation_kwargs)
        prompt_tokens = int(encoded["input_ids"].shape[-1])
        new_ids = output[0, prompt_tokens:]
        return Generation(
            self._tokenizer.decode(new_ids, skip_special_tokens=True).strip(),
            prompt_tokens,
            int(new_ids.shape[-1]),
        )


def build_backend(
    model: str,
    temperature: float = 0.0,
    max_new_tokens: int = 512,
    enable_thinking: bool = False,
) -> ModelBackend:
    if model == "offline-static":
        return StaticBackend()
    if model.startswith("local-transformers:"):
        return TransformersBackend(
            model.split(":", 1)[1],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
        )
    raise ValueError(
        "The paper runner requires local-transformers:<checkpoint directory>; "
        "offline-static is reserved for transport diagnostics."
    )
