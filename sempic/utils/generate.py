import copy
import hashlib
import json
from typing import Protocol, TypedDict, TypeAlias

import torch
from transformers import GenerationConfig
from transformers.generation.utils import (
    GenerateBeamDecoderOnlyOutput,
    GenerateDecoderOnlyOutput,
)
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from ..dataset.abc import SemanticSample
from ..model import SupportedModel
from .lora import get_model_device
from .runtime import get_current_debug_recorder, tensor_summary

TokenizerType: TypeAlias = PreTrainedTokenizer | PreTrainedTokenizerFast


class GenerationOutput(TypedDict):
    sequences: list[torch.Tensor] # (num_seq) [generated_seq_len]
    logits: list[torch.Tensor]    # (num_seq) [generated_seq_len, vocab_size]
    text: list[str]               # (num_seq) strings


class GenerationCacheAccess(Protocol):
    def get(
        self,
        key: str,
        device: torch.device | str | None = None,
    ) -> GenerationOutput | None: ...


class GenerationCache:
    """ Cache for storing generation outputs to avoid redundant computations. """
    def __init__(
        self,
        device: torch.device|None = None,
    ):
        self.cache: dict[str, GenerationOutput] = {}
        self.device = device


    def key_for_sample(self, payload: SemanticSample) -> str:
        return generation_cache_key(payload)


    def get(self, key: str, device: torch.device|None=None) -> GenerationOutput | None:
        generation = self.cache.get(key, None)
        if generation is None:
            return None
        if device is not None and self.device != device:
            generation = GenerationOutput(
                sequences=[seq.to(device) for seq in generation["sequences"]],
                logits=[logit.to(device) for logit in generation["logits"]],
                text=generation["text"],
            )
        return generation


    def add(self, key: str, generation: GenerationOutput):
        """ Add a generation output to the cache. """
        if self.device is not None:
            generation["sequences"] = [seq.to(self.device) for seq in generation["sequences"]]
            generation["logits"] = [logit.to(self.device) for logit in generation["logits"]]
        self.cache[key] = generation


    def __contains__(self, key: str) -> bool:
        """ Check if a generation output is in the cache. """
        return key in self.cache


    def __len__(self) -> int:
        return len(self.cache)


    def keys(self) -> tuple[str, ...]:
        return tuple(self.cache)


def resolve_generation_config(
    model: SupportedModel,
    generation_config: GenerationConfig | None,
) -> GenerationConfig:
    resolved = copy.deepcopy(
        generation_config if generation_config is not None else GenerationConfig()
    )
    model_config = model.generation_config
    resolved.update(
        **model_config.to_dict(),
        defaults_only=True,
        allow_custom_entries=True,
    )
    resolved.update(
        **model_config._get_default_generation_params(),
        defaults_only=True,
    )
    if resolved.cache_implementation == "hybrid":
        resolved.cache_implementation = None
    return resolved


_SEMANTIC_SAMPLE_FIELDS = frozenset({"documents", "query", "shots", "task"})


def generation_cache_key(payload: SemanticSample) -> str:
    if set(payload) != _SEMANTIC_SAMPLE_FIELDS:
        raise ValueError(
            "Generation cache semantic payload must contain exactly these fields: "
            "documents, query, shots, task."
        )
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def get_teacher_logits(
    model: SupportedModel,
    *,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    sequences: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Compute next-token logits for materialized teacher sequences."""
    model_device = get_model_device(model)
    prompt_input_ids = prompt_input_ids.to(model_device)
    prompt_attention_mask = prompt_attention_mask.to(model_device)
    prompt_width = prompt_input_ids.size(1)
    max_target_length = max(sequence.numel() for sequence in sequences)
    full_input_ids = torch.zeros(
        (len(sequences), prompt_width + max_target_length),
        dtype=prompt_input_ids.dtype,
        device=model_device,
    )
    full_attention_mask = torch.zeros_like(full_input_ids)
    full_input_ids[:, :prompt_width] = prompt_input_ids
    full_attention_mask[:, :prompt_width] = prompt_attention_mask
    target_lengths: list[int] = []
    for row, sequence in enumerate(sequences):
        target_length = sequence.numel()
        target_lengths.append(target_length)
        full_input_ids[row, prompt_width:prompt_width + target_length] = sequence.to(model_device)
        full_attention_mask[row, prompt_width:prompt_width + target_length] = 1

    position_ids = full_attention_mask.cumsum(dim=-1) - 1
    position_ids.masked_fill_(full_attention_mask == 0, 1)
    logits_to_keep = torch.arange(
        prompt_width - 1,
        prompt_width - 1 + max_target_length,
        device=model_device,
    )
    with torch.inference_mode():
        teacher_output = model(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            position_ids=position_ids,
            use_cache=False,
            logits_to_keep=logits_to_keep,
        )
    return [
        teacher_output.logits[row, :target_length]
        for row, target_length in enumerate(target_lengths)
    ]


def get_generation(
    model: SupportedModel,
    tokenizer: TokenizerType,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    generation_config: GenerationConfig | None = None,
    output_logits: bool = True,
) -> GenerationOutput:
    """Generate from canonical prompt IDs and an explicit validity mask."""
    assert isinstance(tokenizer.pad_token_id, int)

    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids and attention_mask must have the same two-dimensional shape.")

    model_device = get_model_device(model)
    input_ids = input_ids.to(model_device)
    attention_mask = attention_mask.to(model_device)
    effective_config = resolve_generation_config(model, generation_config)
    effective_config.return_dict_in_generate = True
    use_generation_logits = output_logits and effective_config.num_beams == 1
    effective_config.output_logits = use_generation_logits

    with torch.inference_mode():
        generation_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
            generation_config=effective_config,
        )
    
    assert isinstance(
        generation_output,
        (GenerateDecoderOnlyOutput, GenerateBeamDecoderOnlyOutput),
    )
    sequence = generation_output.sequences # [batch_size, seq_len]
    start_index = input_ids.size(1)

    generation = GenerationOutput(
        sequences=[],
        logits=[],
        text=[]
    )

    for i in range(sequence.size(0)):
        seq = sequence[i][start_index:] # [generated_seq_len]
        eos_token_id = effective_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_mask = seq == eos_token_id
        elif isinstance(eos_token_id, list):
            eos_mask = torch.zeros_like(seq, dtype=torch.bool)
            for token_id in eos_token_id:
                eos_mask |= seq == token_id
        else:
            eos_mask = torch.zeros_like(seq, dtype=torch.bool)
        eos_indices = eos_mask.nonzero()
        if eos_indices.numel() > 0:
            end_index = int(eos_indices[0].item()) + 1
            seq = seq[:end_index]

        text = tokenizer.decode(seq, skip_special_tokens=False)
        assert isinstance(text, str)
        generation["sequences"].append(seq)
        generation["text"].append(text)
    if use_generation_logits:
        if generation_output.logits is None:
            raise RuntimeError("Teacher generation did not return requested step logits.")
        step_logits = generation_output.logits
        if len(step_logits) < max(sequence.numel() for sequence in generation["sequences"]):
            raise RuntimeError("Teacher generation returned fewer logits than generated tokens.")
        teacher_dtype = next(model.parameters()).dtype
        generation["logits"] = [
            torch.stack([
                step_logits[step][row]
                for step in range(sequence.numel())
            ]).to(dtype=teacher_dtype)
            for row, sequence in enumerate(generation["sequences"])
        ]
    elif output_logits:
        generation["logits"] = get_teacher_logits(
            model,
            prompt_input_ids=input_ids.repeat_interleave(
                effective_config.num_return_sequences,
                dim=0,
            ),
            prompt_attention_mask=attention_mask.repeat_interleave(
                effective_config.num_return_sequences,
                dim=0,
            ),
            sequences=generation["sequences"],
        )
    return generation


def get_answers(
    generated_tokens: torch.Tensor,
    input_ids: torch.Tensor,
    tokenizer: TokenizerType
) -> list[str]:
    """ Extract answers from generated tokens based on input IDs and tokenizer. """
    num_seq = input_ids.size(0)
    start_index = input_ids.size(1)
    answers: list[str] = []
    for i in range(num_seq):
        generation = generated_tokens[i][start_index:]
        eos_indices = (generation == tokenizer.eos_token_id).nonzero() # type: ignore
        if eos_indices.numel() > 0:
            end_index = eos_indices[0].item()
            generation = generation[:end_index]
        generated_text = tokenizer.decode(generation, skip_special_tokens=True)
        assert isinstance(generated_text, str)
        answers.append(generated_text)
    debug_recorder = get_current_debug_recorder()
    if debug_recorder is not None and debug_recorder.enabled:
        generation_start = input_ids.size(1)
        generated_only = generated_tokens[:, generation_start:]
        debug_recorder.record_json(
            "generation_answers",
            {
                "input_ids": debug_recorder.token_ids(input_ids),
                "generated_tokens": (
                    debug_recorder.token_ids(generated_only)
                    if debug_recorder.config["save_token_ids"]
                    else tensor_summary(generated_only)
                ),
                "answers": [
                    {"text": answer, "length": len(answer)}
                    for answer in answers
                ],
            },
        )
    return answers
