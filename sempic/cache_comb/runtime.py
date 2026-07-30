from contextlib import contextmanager
from time import perf_counter
from types import MethodType
from typing import Iterator

import torch
from transformers import GenerationConfig
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..model import SupportedModel
from .abc import PrefillResult, TokenizerType


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class TTFTTimer(StoppingCriteria):
    def __init__(self, device: torch.device):
        self.device = device
        self.started_at: float | None = None
        self.elapsed: float | None = None

    def start(self) -> None:
        if self.started_at is not None:
            raise RuntimeError("TTFT timer has already started.")
        synchronize(self.device)
        self.started_at = perf_counter()

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs: object,
    ) -> torch.BoolTensor:
        if self.started_at is None:
            raise RuntimeError("TTFT timer must start before generation.")
        if self.elapsed is None:
            synchronize(input_ids.device)
            self.elapsed = perf_counter() - self.started_at
        return torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)

    def result(self) -> float:
        if self.elapsed is None:
            raise ValueError("Generation produced no first token; max_new_tokens must be positive.")
        return self.elapsed


@contextmanager
def _inject_prefill(
    model: SupportedModel,
    result: PrefillResult,
) -> Iterator[None]:
    had_instance_prefill = "_prefill" in model.__dict__
    original = model.__dict__.get("_prefill")

    def injected_prefill(
        self: SupportedModel,
        input_ids: torch.LongTensor,
        generation_config: GenerationConfig,
        model_kwargs: dict,
    ) -> CausalLMOutputWithPast:
        logits = result.logits
        if logits.size(0) != input_ids.size(0):
            if input_ids.size(0) % logits.size(0) != 0:
                raise ValueError("Generation expanded to an incompatible batch size.")
            repeats = input_ids.size(0) // logits.size(0)
            logits = logits.repeat_interleave(repeats, dim=0)
            model_kwargs["past_key_values"].batch_repeat_interleave(repeats)
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=model_kwargs["past_key_values"],
        )

    model._prefill = MethodType(injected_prefill, model)
    try:
        yield
    finally:
        if had_instance_prefill:
            model._prefill = original
        else:
            del model._prefill


def generate_from_prefill(
    *,
    model: SupportedModel,
    tokenizer: TokenizerType,
    generation_config: GenerationConfig | None,
    result: PrefillResult,
    ttft_timer: TTFTTimer,
) -> tuple[torch.Tensor, float]:
    cache_len = result.past_key_values.get_seq_length()
    cache_position = torch.arange(
        cache_len, dtype=torch.long, device=result.generation_input_ids.device
    )
    with _inject_prefill(model, result), torch.no_grad():
        generation = model.generate(
            input_ids=result.generation_input_ids,
            attention_mask=result.attention_mask,
            position_ids=result.position_ids,
            past_key_values=result.past_key_values,
            cache_position=cache_position,
            generation_config=generation_config,
            tokenizer=tokenizer,
            stopping_criteria=StoppingCriteriaList([ttft_timer]),
        )
    if isinstance(generation, torch.Tensor):
        sequences = generation
    else:
        sequences = generation.sequences
    return sequences, ttft_timer.result()
