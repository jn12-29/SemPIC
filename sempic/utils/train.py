import gc
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Literal, TypedDict, cast
from warnings import warn

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import GenerationConfig

from ..model import SupportedModel
from ..packet_wrapper import PacketWrapper
from ..prompt import TokenizedPrompt
from .generate import (
    GenerationCache,
    GenerationCacheAccess,
    GenerationOutput,
    TokenizerType,
    get_generation,
    get_teacher_logits,
    resolve_generation_config,
)
from .lora import (
    disable_lora_adapters,
    get_model_device,
)
from .runtime import (
    DebugDumpConfig,
    DebugRecorder,
    LoggingConfig,
    generation_summary,
    load_debug_dump_config,
    load_logging_config,
    tensor_summary,
)
from .run_storage import validate_run_suffix
from .student_prefill import (
    TrainAttentionBackend,
    batched_student_loss,
    probe_train_flex_attention_shapes,
)


LOGGER = logging.getLogger(__name__)

TrainTarget = Literal["lora", "packet_wrapper"]
LossType = Literal["kl", "ce"]
TrainAttentionPreference = Literal["auto", "flex", "sdpa"]
TARGETS: tuple[TrainTarget, ...] = ("lora", "packet_wrapper")


@dataclass(slots=True)
class TrainAttentionRuntime:
    backend: TrainAttentionBackend
    verified: bool


def create_train_attention_runtime(
    model: Any,
    forward_batch_size: int,
    attention_backend: TrainAttentionPreference = "auto",
) -> TrainAttentionRuntime:
    if attention_backend == "sdpa":
        return TrainAttentionRuntime(backend="sdpa", verified=True)
    device = get_model_device(model)
    if device.type != "cuda":
        if attention_backend == "flex":
            LOGGER.warning(
                "Train FlexAttention requires CUDA; falling back to SDPA on %s.",
                device,
            )
        return TrainAttentionRuntime(backend="sdpa", verified=True)
    if forward_batch_size != 1:
        LOGGER.warning(
            "Train FlexAttention currently requires forward_batch_size=1; "
            "falling back to SDPA for forward_batch_size=%d.",
            forward_batch_size,
        )
        return TrainAttentionRuntime(backend="sdpa", verified=True)
    return TrainAttentionRuntime(backend="flex", verified=False)


class TrainSample(TypedDict):
    prompt: TokenizedPrompt
    semantic_key: str


class ModelConfig(TypedDict):
    model_path: str
    dtype: str
    device: str
    generation_kwargs: dict


class DatasetConfig(TypedDict):
    dataset_name: str
    num_samples: int
    num_data_strs: int
    num_shots: int
    subset: str
    split: str
    seed: int
    data_kwargs: dict
    template: str
    template_kwargs: dict


class TrainSettings(TypedDict):
    targets: list[TrainTarget]


class LossConfig(TypedDict):
    type: LossType
    tau: float


class LoRAConfigDict(TypedDict):
    enabled: bool
    rank: int
    alpha: int
    dropout: float
    target_modules: list[str]
    adapter_name: str
    init_path: str | None


class PacketWrapperConfig(TypedDict):
    enabled: bool
    header_len: int | None
    trailer_len: int | None
    dtype: str | None
    init_path: str | None


class OptimizerConfig(TypedDict):
    opt_config: dict
    scheduler_config: dict


class TrainConfig(TypedDict):
    output_dir: str
    run_suffix: str | None
    total_epoch: int
    gen_batch_size: int
    batch_size: int
    forward_batch_size: int
    attention_backend: TrainAttentionPreference
    use_cache: bool
    model: ModelConfig
    cache_device: str
    cache_path: list[str] | None
    kv_gradient_checkpointing: bool
    seed: int
    train: TrainSettings
    loss: LossConfig
    lora: LoRAConfigDict
    packet_wrapper: PacketWrapperConfig
    optimizers: dict[TrainTarget, OptimizerConfig]
    data_configs: list[DatasetConfig]
    logging: LoggingConfig
    debug_dump: DebugDumpConfig


dtype_map: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "double": torch.double,
    "int64": torch.int64,
    "long": torch.long,
    "bool": torch.bool,
}


LEGACY_TOP_LEVEL_FIELDS = {
    "ckpt" + "_epoch",
    "dis" + "till",
    "file_name",
    "header_len",
    "lora" + "_adapter",
    "opt_config",
    "res" + "ume",
    "res" + "ume_epoch",
    "save_path",
    "scheduler_config",
    "trailer_len",
    "use" + "_logits",
}


def _reject_legacy_fields(config: dict[str, Any]) -> None:
    legacy_fields = sorted(LEGACY_TOP_LEVEL_FIELDS.intersection(config))
    if legacy_fields:
        raise ValueError(
            "Unsupported legacy training fields: "
            + ", ".join(legacy_fields)
            + ". Use lora.*, packet_wrapper.*, loss.*, train.targets, and optimizers.*."
        )


def _as_bool(config: dict[str, Any], key: str, default: bool = False) -> bool:
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean.")
    return value


def _as_target(value: str) -> TrainTarget:
    if value not in TARGETS:
        raise ValueError(f"Unsupported train target: {value}")
    return cast(TrainTarget, value)


def _load_data_configs(config: dict[str, Any]) -> list[DatasetConfig]:
    data_configs: list[DatasetConfig] = []
    for data_conf in config["data_configs"]:
        data_configs.append(DatasetConfig(
            dataset_name=data_conf["dataset_name"],
            num_data_strs=data_conf["num_data_strs"],
            num_shots=data_conf["num_shots"],
            num_samples=data_conf["num_samples"],
            subset=data_conf["subset"],
            split=data_conf.get("split", "train"),
            seed=data_conf.get("seed", 42),
            data_kwargs=data_conf.get("data_kwargs", {}),
            template=data_conf.get("template", ""),
            template_kwargs=data_conf.get("template_kwargs", {}),
        ))
    return data_configs


def _load_cache_paths(value: Any) -> list[str] | None:
    if value is None:
        return None
    paths = [value] if isinstance(value, str) else value
    if not isinstance(paths, list) or not paths:
        raise ValueError("cache_path must be null, a non-empty string, or a non-empty list.")
    if any(not isinstance(path, str) or not path for path in paths):
        raise ValueError("cache_path entries must be non-empty strings.")
    if len(set(paths)) != len(paths):
        raise ValueError("cache_path must not contain duplicate paths.")
    return list(paths)


def load_train_config(config: dict[str, Any]) -> TrainConfig:
    _reject_legacy_fields(config)

    output_dir = config.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir:
        raise ValueError("output_dir must be a non-empty string.")
    run_suffix = config.get("run_suffix")
    if run_suffix is not None and not isinstance(run_suffix, str):
        raise ValueError("run_suffix must be a string or null.")
    validate_run_suffix(run_suffix)

    model = ModelConfig(
        model_path=config["model"]["model_path"],
        dtype=config["model"].get("dtype", "bfloat16"),
        device=config["model"].get("device", "cuda:0"),
        generation_kwargs=config["model"].get("generation_kwargs", {}),
    )

    lora_config = config.get("lora", {})
    if "save_path" in lora_config:
        raise ValueError("lora.save_path is unsupported. Use output_dir.")
    lora_enabled = _as_bool(lora_config, "enabled")
    adapter_name = lora_config.get("adapter_name", "default")
    if lora_enabled and adapter_name != "default":
        raise ValueError("lora.adapter_name must be 'default'.")
    lora = LoRAConfigDict(
        enabled=lora_enabled,
        rank=lora_config.get("rank", 8),
        alpha=lora_config.get("alpha", 16),
        dropout=lora_config.get("dropout", 0.0),
        target_modules=lora_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        adapter_name=adapter_name,
        init_path=lora_config.get("init_path", None),
    )

    packet_config = config.get("packet_wrapper", {})
    legacy_packet_fields = sorted({"save_path", "file_name"}.intersection(packet_config))
    if legacy_packet_fields:
        raise ValueError(
            "Unsupported packet_wrapper output fields: "
            + ", ".join(legacy_packet_fields)
            + ". Use output_dir."
        )
    packet_enabled = _as_bool(packet_config, "enabled")
    packet_wrapper = PacketWrapperConfig(
        enabled=packet_enabled,
        header_len=packet_config.get("header_len", None),
        trailer_len=packet_config.get("trailer_len", None),
        dtype=packet_config.get("dtype", None),
        init_path=packet_config.get("init_path", None),
    )

    enabled_targets: list[TrainTarget] = []
    if lora_enabled:
        enabled_targets.append("lora")
    if packet_enabled:
        enabled_targets.append("packet_wrapper")
    if not enabled_targets:
        raise ValueError("At least one component must be enabled.")

    train_config = config.get("train", {})
    raw_targets = train_config.get("targets", enabled_targets)
    if not isinstance(raw_targets, list) or len(raw_targets) == 0:
        raise ValueError("train.targets must be a non-empty list.")
    targets = [_as_target(target) for target in raw_targets]
    if len(set(targets)) != len(targets):
        raise ValueError("train.targets must not contain duplicates.")
    disabled_targets = [target for target in targets if target not in enabled_targets]
    if disabled_targets:
        raise ValueError(
            "train.targets can only include enabled components: "
            + ", ".join(disabled_targets)
        )

    if packet_enabled and packet_wrapper["init_path"] is None:
        if packet_wrapper["header_len"] is None or packet_wrapper["trailer_len"] is None:
            raise ValueError(
                "packet_wrapper.header_len and packet_wrapper.trailer_len are required "
                "when packet_wrapper.init_path is not set."
            )

    loss_config = config.get("loss", {})
    loss_kind = loss_config.get("type", "kl")
    if loss_kind not in ("kl", "ce"):
        raise ValueError("loss.type must be 'kl' or 'ce'.")
    tau = float(loss_config.get("tau", 1.0))
    if loss_kind == "kl" and tau <= 0:
        raise ValueError("loss.tau must be positive for KL loss.")
    loss = LossConfig(type=cast(LossType, loss_kind), tau=tau)

    kv_gradient_checkpointing = _as_bool(
        config,
        "kv_gradient_checkpointing",
        default=True,
    )

    optimizer_configs: dict[TrainTarget, OptimizerConfig] = {}
    optimizers = config.get("optimizers", {})
    for target in targets:
        target_optimizer = optimizers.get(target, None)
        if target_optimizer is None:
            raise ValueError(f"optimizers.{target} is required when training {target}.")
        optimizer_configs[target] = OptimizerConfig(
            opt_config=target_optimizer["opt_config"],
            scheduler_config=target_optimizer.get("scheduler_config", {}),
        )

    forward_batch_size = int(config.get("forward_batch_size", 1))
    attention_backend = config.get("attention_backend", "auto")
    if attention_backend not in ("auto", "flex", "sdpa"):
        raise ValueError("attention_backend must be 'auto', 'flex', or 'sdpa'.")
    batch_size = int(config["batch_size"])
    if forward_batch_size <= 0:
        raise ValueError("forward_batch_size must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if batch_size % forward_batch_size != 0:
        raise ValueError("forward_batch_size must divide batch_size.")

    cache_paths = _load_cache_paths(config.get("cache_path"))
    cache_device = config.get("cache_device", "cuda:0")
    if cache_paths is not None and torch.device(cache_device).type != "cpu":
        raise ValueError("Configured generation caches require cache_device to be CPU.")

    return TrainConfig(
        output_dir=output_dir,
        run_suffix=run_suffix,
        total_epoch=config["total_epoch"],
        gen_batch_size=config["gen_batch_size"],
        batch_size=batch_size,
        forward_batch_size=forward_batch_size,
        attention_backend=cast(TrainAttentionPreference, attention_backend),
        use_cache=config.get("use_cache", False),
        model=model,
        cache_device=cache_device,
        cache_path=cache_paths,
        kv_gradient_checkpointing=kv_gradient_checkpointing,
        seed=config.get("seed", 42),
        train=TrainSettings(targets=targets),
        loss=loss,
        lora=lora,
        packet_wrapper=packet_wrapper,
        optimizers=optimizer_configs,
        data_configs=_load_data_configs(config),
        logging=load_logging_config(config),
        debug_dump=load_debug_dump_config(config),
    )


def _left_pad_prompt_ids(
    prompts: list[TokenizedPrompt],
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_length = max(prompt.input_ids.numel() for prompt in prompts)
    input_ids = torch.full(
        (len(prompts), max_length),
        fill_value=pad_token_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, prompt in enumerate(prompts):
        length = prompt.input_ids.numel()
        input_ids[row, max_length - length:] = prompt.input_ids
        attention_mask[row, max_length - length:] = 1
    return input_ids, attention_mask


def build_generation_cache(
    samples: list[TrainSample],
    batch_size: int,
    model: SupportedModel,
    tokenizer: TokenizerType,
    generation_config: GenerationConfig | None = None,
    generation_cache: GenerationCache | None = None,
    store_logits: bool = True,
    debug_recorder: DebugRecorder | None = None,
    generation_sink: Callable[[str, GenerationOutput], None] | None = None,
) -> tuple[GenerationCache, bool]:
    if generation_cache is None:
        generation_cache = GenerationCache()

    effective_config = resolve_generation_config(model, generation_config)

    samples_to_gen_by_key: dict[str, TrainSample] = {}
    samples_to_refresh_by_key: dict[str, tuple[TrainSample, Any]] = {}
    for sample in samples:
        key = sample["semantic_key"]
        generation = generation_cache.get(key)
        if generation is None:
            samples_to_gen_by_key.setdefault(key, sample)
        elif store_logits and len(generation["logits"]) != len(generation["sequences"]):
            samples_to_refresh_by_key.setdefault(key, (sample, generation))

    for key, (sample, generation) in samples_to_refresh_by_key.items():
        prompt_ids = sample["prompt"].input_ids.unsqueeze(0)
        num_sequences = len(generation["sequences"])
        logits = get_teacher_logits(
            model,
            prompt_input_ids=prompt_ids.repeat(num_sequences, 1),
            prompt_attention_mask=torch.ones_like(prompt_ids).repeat(num_sequences, 1),
            sequences=generation["sequences"],
        )
        generation["logits"] = (
            [logits_tensor.to(generation_cache.device) for logits_tensor in logits]
            if generation_cache.device is not None
            else logits
        )
    changed = bool(samples_to_refresh_by_key)
    samples_to_gen = [
        (sample, key)
        for key, sample in samples_to_gen_by_key.items()
    ]

    if len(samples_to_gen) == 0:
        if debug_recorder is not None:
            debug_recorder.child("generation_cache").record_json(
                "status",
                {
                    "status": "hit",
                    "num_cached": len(generation_cache),
                    "store_logits": store_logits,
                },
            )
        return generation_cache, changed

    if batch_size <= 0:
        batch_size = len(samples_to_gen)

    cache_debug = (
        debug_recorder.child("generation_cache")
        if debug_recorder is not None
        else None
    )
    if cache_debug is not None:
        cache_debug.record_json(
            "status",
            {
                "status": "building",
                "num_cached_before": len(generation_cache),
                "num_to_generate": len(samples_to_gen),
                "batch_size": batch_size,
                "store_logits": store_logits,
                "input_lengths": [
                    item[0]["prompt"].input_ids.numel()
                    for item in samples_to_gen
                ],
            },
        )
    for i in range(0, len(samples_to_gen), batch_size):
        batch = samples_to_gen[i: i + batch_size]
        batch_prompts = [sample["prompt"] for sample, _ in batch]
        assert isinstance(tokenizer.pad_token_id, int)
        batch_input_ids, batch_attention_mask = _left_pad_prompt_ids(
            batch_prompts,
            tokenizer.pad_token_id,
        )
        gen = get_generation(
            model,
            tokenizer,
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            generation_config=effective_config,
            output_logits=store_logits,
        )
        num_return_sequences = effective_config.num_return_sequences

        for k in range(len(batch)):
            sample_index = i + k
            sample, key = samples_to_gen[sample_index]
            start = k * num_return_sequences
            end = start + num_return_sequences
            generation = {
                "sequences": gen["sequences"][start:end],
                "logits": gen["logits"][start:end] if store_logits else [],
                "text": gen["text"][start:end],
            }
            sample_debug = (
                cache_debug.sample_scope("sample", sample_index)
                if cache_debug is not None
                else None
            )
            if sample_debug is not None and not sample_debug.enabled:
                sample_debug = None
            if sample_debug is not None:
                sample_debug.record_json(
                    "generation",
                    {
                        "input_length": sample["prompt"].input_ids.numel(),
                        "generation": generation_summary(
                            generation,
                            save_token_ids=sample_debug.config["save_token_ids"],
                        ),
                        "stored_logits": store_logits,
                        "input_tensor": [
                            tensor_summary(sequence)
                            for sequence in generation["sequences"]
                        ],
                    },
                )
            if generation_sink is None:
                generation_cache.add(key, generation)
            else:
                generation_sink(key, generation)
            changed = True

    return generation_cache, changed


def build_peft_lora_model(
    base_model: SupportedModel,
    lora_config: LoRAConfigDict,
) -> Any:
    peft_config = LoraConfig(
        r=lora_config["rank"],
        lora_alpha=lora_config["alpha"],
        target_modules=lora_config["target_modules"],
        lora_dropout=lora_config["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(base_model, peft_config, adapter_name=lora_config["adapter_name"])


def load_or_build_lora_model(
    base_model: SupportedModel,
    lora_config: LoRAConfigDict,
) -> Any:
    if lora_config["init_path"] is not None:
        return PeftModel.from_pretrained(
            base_model,
            lora_config["init_path"],
            adapter_name=lora_config["adapter_name"],
            is_trainable=True,
        )
    return build_peft_lora_model(base_model, lora_config)


def save_lora_adapter(model: Any, save_path: str, adapter_name: str | None = None) -> None:
    if adapter_name is None:
        model.save_pretrained(save_path)
        return

    try:
        model.save_pretrained(save_path, selected_adapters=[adapter_name])
    except TypeError:
        model.save_pretrained(save_path)


def configure_trainable_parameters(
    model: Any,
    packet_wrapper: PacketWrapper | None,
    targets: list[TrainTarget],
) -> None:
    for _, param in model.named_parameters():
        param.requires_grad = False

    if "lora" in targets:
        lora_param_count = 0
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
                lora_param_count += 1
        if lora_param_count == 0:
            raise ValueError("No LoRA parameters are trainable. Check target_modules and adapter loading.")

    if packet_wrapper is not None:
        packet_requires_grad = "packet_wrapper" in targets
        packet_wrapper.header.requires_grad_(packet_requires_grad)
        packet_wrapper.trailer.requires_grad_(packet_requires_grad)


def target_parameters(
    model: Any,
    packet_wrapper: PacketWrapper | None,
    target: TrainTarget,
) -> list[torch.Tensor]:
    if target == "lora":
        return [
            param for name, param in model.named_parameters()
            if "lora_" in name and param.requires_grad
        ]
    if packet_wrapper is None:
        return []
    return [
        param for param in (packet_wrapper.header, packet_wrapper.trailer)
        if param.requires_grad
    ]


def _zero_optimizers(optimizers: dict[TrainTarget, torch.optim.Optimizer]) -> None:
    for optimizer in optimizers.values():
        optimizer.zero_grad()


def _step_optimizers(
    optimizers: dict[TrainTarget, torch.optim.Optimizer],
    schedulers: dict[TrainTarget, torch.optim.lr_scheduler.LRScheduler],
) -> dict[TrainTarget, float]:
    lrs: dict[TrainTarget, float] = {}
    for target, optimizer in optimizers.items():
        optimizer.step()
        schedulers[target].step()
        lrs[target] = schedulers[target].get_last_lr()[0]
    _zero_optimizers(optimizers)
    return lrs


def _format_lrs(lrs: dict[TrainTarget, float]) -> str:
    return "{" + ", ".join(
        f"{target!r}: {lr:.4g}" for target, lr in lrs.items()
    ) + "}"


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _verify_train_attention_backend(
    runtime: TrainAttentionRuntime,
    *,
    samples: list[TrainSample],
    first_sample: TrainSample,
    model: Any,
    optimizers: dict[TrainTarget, torch.optim.Optimizer],
    generation_cache: GenerationCacheAccess,
    loss_config: LossConfig,
    lora_enabled: bool,
    lora_adapter_name: str | None,
    packet_wrapper: PacketWrapper | None,
    kv_gradient_checkpointing: bool,
) -> None:
    if runtime.verified:
        return

    device = get_model_device(model)
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = (
        torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    )

    def restore_probe_state() -> None:
        _zero_optimizers(optimizers)
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state, device)

    def probe(backend: TrainAttentionBackend) -> None:
        loss, _, _ = batched_student_loss(
            samples=[first_sample],
            model=model,
            generation_cache=generation_cache,
            loss_config=loss_config,
            lora_enabled=lora_enabled,
            lora_adapter_name=lora_adapter_name,
            packet_wrapper=packet_wrapper,
            kv_gradient_checkpointing=kv_gradient_checkpointing,
            attention_backend=backend,
        )
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    try:
        probe_train_flex_attention_shapes(
            samples,
            model,
            generation_cache,
            packet_wrapper,
        )
        probe("flex")
    except Exception as flex_error:
        flex_error_summary = f"{type(flex_error).__name__}: {flex_error}"
        flex_error.__traceback__ = None
        restore_probe_state()
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        try:
            probe("sdpa")
        except Exception as sdpa_error:
            restore_probe_state()
            raise RuntimeError(
                "Both Train FlexAttention and its SDPA fallback failed during "
                "the preflight backward probe. "
                f"FlexAttention: {flex_error_summary}; "
                f"SDPA: {type(sdpa_error).__name__}: {sdpa_error}"
            ) from sdpa_error
        restore_probe_state()
        runtime.backend = "sdpa"
        runtime.verified = True
        LOGGER.warning(
            "Train FlexAttention preflight failed; falling back to SDPA for "
            "this training process: %s",
            flex_error_summary,
        )
        return

    restore_probe_state()
    runtime.verified = True
    LOGGER.info("Train FlexAttention forward/backward preflight succeeded.")


def train_components(
    samples: list[TrainSample],
    model: Any,
    batch_size: int,
    forward_batch_size: int,
    optimizers: dict[TrainTarget, torch.optim.Optimizer],
    schedulers: dict[TrainTarget, torch.optim.lr_scheduler.LRScheduler],
    generation_cache: GenerationCacheAccess,
    loss_config: LossConfig,
    lora_enabled: bool,
    lora_adapter_name: str | None,
    packet_wrapper: PacketWrapper | None,
    kv_gradient_checkpointing: bool = False,
    epoch: int = -1,
    total_epochs: int | None = None,
    epoch_indices: list[int] | None = None,
    debug_recorder: DebugRecorder | None = None,
    attention_runtime: TrainAttentionRuntime | None = None,
) -> GenerationCacheAccess:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if forward_batch_size <= 0:
        raise ValueError("forward_batch_size must be positive.")
    if batch_size % forward_batch_size != 0:
        raise ValueError("forward_batch_size must divide batch_size.")
    if len(optimizers) == 0:
        raise ValueError("At least one optimizer is required.")

    if lora_enabled:
        disable_lora_adapters(model)
        model.train()
    else:
        model.eval()

    sample_debug_recorder = debug_recorder if epoch <= 0 else None
    train_step = 0
    eval_tokens = 0
    acc_loss = 0.0
    epoch_eval_tokens = 0
    epoch_acc_loss = 0.0
    optimizer_step_idx = 0

    if epoch_indices is None:
        epoch_indices = list(range(len(samples)))
    else:
        assert len(epoch_indices) == len(samples)
    total_optimizer_steps = (len(epoch_indices) + batch_size - 1) // batch_size

    if attention_runtime is not None and epoch_indices:
        _verify_train_attention_backend(
            attention_runtime,
            samples=samples,
            first_sample=samples[epoch_indices[0]],
            model=model,
            optimizers=optimizers,
            generation_cache=generation_cache,
            loss_config=loss_config,
            lora_enabled=lora_enabled,
            lora_adapter_name=lora_adapter_name,
            packet_wrapper=packet_wrapper,
            kv_gradient_checkpointing=kv_gradient_checkpointing,
        )
    attention_backend: TrainAttentionBackend = (
        attention_runtime.backend if attention_runtime is not None else "sdpa"
    )

    _zero_optimizers(optimizers)
    model_device = get_model_device(model)
    epoch_started_at = time.perf_counter()
    for batch_start in range(0, len(epoch_indices), batch_size):
        step_started_at = time.perf_counter()
        optimizer_indices = epoch_indices[batch_start:batch_start + batch_size]
        for micro_start in range(0, len(optimizer_indices), forward_batch_size):
            micro_indices = optimizer_indices[micro_start:micro_start + forward_batch_size]
            micro_samples = [samples[index] for index in micro_indices]
            micro_debuggers: list[DebugRecorder | None] = []
            for offset in range(len(micro_samples)):
                sample_debug = (
                    sample_debug_recorder.sample_scope("train_sample", train_step + offset)
                    if sample_debug_recorder is not None
                    else None
                )
                if sample_debug is not None and not sample_debug.enabled:
                    sample_debug = None
                micro_debuggers.append(sample_debug)

            loss, num_tokens, sample_loss_values = batched_student_loss(
                samples=micro_samples,
                model=model,
                generation_cache=generation_cache,
                loss_config=loss_config,
                lora_enabled=lora_enabled,
                lora_adapter_name=lora_adapter_name,
                packet_wrapper=packet_wrapper,
                kv_gradient_checkpointing=kv_gradient_checkpointing,
                debug_recorders=micro_debuggers,
                attention_backend=attention_backend,
            )
            loss.backward()

            loss_value = float(loss.detach().item())
            acc_loss += loss_value
            eval_tokens += num_tokens
            epoch_acc_loss += loss_value
            epoch_eval_tokens += num_tokens

            if sample_loss_values is not None:
                for offset, (sample_index, sample_loss_value) in enumerate(
                    zip(micro_indices, sample_loss_values, strict=True)
                ):
                    sample_debug = micro_debuggers[offset]
                    if sample_debug is not None:
                        sample_debug.record_json(
                            "loss",
                            {
                                "epoch": epoch,
                                "sample_index": sample_index,
                                "train_step": train_step + offset + 1,
                                "loss": sample_loss_value,
                                "microbatch_size": len(micro_samples),
                                "forward_batch_size": forward_batch_size,
                                "batch_size": batch_size,
                            },
                        )

            train_step += len(micro_samples)

        lrs = _step_optimizers(optimizers, schedulers)
        optimizer_step_idx += 1
        if model_device.type == "cuda":
            torch.cuda.synchronize(model_device)
        step_finished_at = time.perf_counter()
        step_seconds = max(step_finished_at - step_started_at, 1e-9)
        epoch_elapsed_seconds = step_finished_at - epoch_started_at
        average_step_seconds = epoch_elapsed_seconds / optimizer_step_idx
        if epoch >= 0 and total_epochs is not None:
            overall_step = epoch * total_optimizer_steps + optimizer_step_idx
            overall_total_steps = total_epochs * total_optimizer_steps
            progress = (
                f"Epoch {epoch + 1}/{total_epochs} | Step {optimizer_step_idx}/{total_optimizer_steps} | "
                f"Overall {overall_step}/{overall_total_steps} "
                f"({100.0 * overall_step / overall_total_steps:.1f}%)"
            )
            remaining_steps = overall_total_steps - overall_step
        else:
            progress = f"Epoch {epoch + 1} | Step {optimizer_step_idx}/{total_optimizer_steps}"
            remaining_steps = total_optimizer_steps - optimizer_step_idx
        eta_seconds = average_step_seconds * remaining_steps
        estimated_finish = datetime.now().astimezone() + timedelta(seconds=eta_seconds)
        LOGGER.info(
            "%s | step_loss %.4g | epoch_avg_loss %.4g | tokens %d | "
            "%.1f tok/s | %.1f samples/s | lr %s | step %.2fs | "
            "epoch_elapsed %s | ETA %s | finish %s",
            progress,
            acc_loss / eval_tokens,
            epoch_acc_loss / epoch_eval_tokens,
            eval_tokens,
            eval_tokens / step_seconds,
            len(optimizer_indices) / step_seconds,
            _format_lrs(lrs),
            step_seconds,
            _format_duration(epoch_elapsed_seconds),
            _format_duration(eta_seconds),
            estimated_finish.isoformat(sep=" ", timespec="seconds"),
        )
        if debug_recorder is not None and debug_recorder.enabled:
            debug_recorder.record_json(
                f"epoch_{epoch}_optimizer_step_{optimizer_step_idx}",
                {
                    "epoch": epoch,
                    "optimizer_step_idx": optimizer_step_idx,
                    "train_step": train_step,
                    "eval_tokens": eval_tokens,
                    "loss_per_token": acc_loss / eval_tokens,
                    "lrs": lrs,
                    "forward_batch_size": forward_batch_size,
                    "partial_final_batch": len(optimizer_indices) < batch_size,
                },
            )
        eval_tokens = 0
        acc_loss = 0.0
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if train_step % batch_size != 0:
        warn("Number of samples is not divisible by batch_size; stepped optimizers on the final partial batch.")

    if epoch_eval_tokens > 0:
        epoch_label = f"epoch {epoch + 1}" if epoch >= 0 else "epoch"
        LOGGER.info(
            "%s eval tokens %d, epoch loss %.4g",
            epoch_label,
            epoch_eval_tokens,
            epoch_acc_loss / epoch_eval_tokens,
        )

    return generation_cache
