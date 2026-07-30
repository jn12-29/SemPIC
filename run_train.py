import argparse
import copy
import gc
import logging
import os
import random
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from sempic.dataset import get_ret_eval_generator
from sempic.model import SupportedModel
from sempic.packet_wrapper import PacketWrapper, WrapperStateDict, load_wrapper
from sempic.prompt import compile_prompt
from sempic.utils.config import gather_config_files, load_config_file
from sempic.utils.generate import (
    GenerationCache,
    GenerationCacheAccess,
    TokenizerType,
    generation_cache_key,
)
from sempic.utils.generation_cache import load_generation_cache
from sempic.utils.lora import disable_lora_adapters, get_model_device, lora_adapters_disabled
from sempic.utils.runtime import (
    DebugRecorder,
    RuntimeCLIOverrides,
    RuntimeContext,
    add_runtime_cli_args,
    apply_runtime_overrides,
    runtime_overrides_from_args,
)
from sempic.utils.run_storage import (
    allocate_run_dir,
    select_existing_run_dir,
)
from sempic.utils.train import (
    TrainConfig,
    TrainSample,
    TrainTarget,
    create_train_attention_runtime,
    build_generation_cache,
    configure_trainable_parameters,
    dtype_map,
    load_or_build_lora_model,
    load_train_config,
    save_lora_adapter,
    target_parameters,
    train_components,
)
from sempic.utils.train_checkpoint import (
    checkpoint_path,
    load_training_checkpoint,
    restore_training_state,
    save_training_checkpoint,
)


LOGGER = logging.getLogger("sempic.run_train")


class _LogFloat(float):
    def __repr__(self) -> str:
        return f"{float(self):.4g}"


def _format_log_floats(value: Any) -> Any:
    if isinstance(value, float):
        return _LogFloat(value)
    if isinstance(value, dict):
        return {key: _format_log_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_format_log_floats(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_format_log_floats(item) for item in value)
    return value


class TrainCache:
    def __init__(self) -> None:
        self.tokenizer_cache: dict[str, TokenizerType] = {}
        self.model_cache: dict[tuple[Any, ...], Any] = {}


def load_tokenizer(train_config: TrainConfig, train_cache: TrainCache) -> TokenizerType:
    model_path = train_config["model"]["model_path"]
    if train_config["use_cache"] and model_path in train_cache.tokenizer_cache:
        return train_cache.tokenizer_cache[model_path]

    tokenizer: TokenizerType = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if train_config["use_cache"]:
        train_cache.tokenizer_cache[model_path] = tokenizer
    return tokenizer


def load_model(train_config: TrainConfig, train_cache: TrainCache) -> Any:
    model_config = train_config["model"]
    lora_config = train_config["lora"]
    model_key = (
        model_config["model_path"],
        model_config["dtype"],
        model_config["device"],
        lora_config["enabled"],
        lora_config["init_path"],
        lora_config["rank"],
        lora_config["alpha"],
        lora_config["dropout"],
        tuple(lora_config["target_modules"]),
        lora_config["adapter_name"],
    )

    if train_config["use_cache"] and model_key in train_cache.model_cache:
        model = train_cache.model_cache[model_key]
        configure_trainable_parameters(
            model,
            None,
            [target for target in train_config["train"]["targets"] if target == "lora"],
        )
        if lora_config["enabled"]:
            disable_lora_adapters(model)
        return model

    device_map: torch.device | str
    if model_config["device"] == "auto":
        device_map = "auto"
    else:
        device_map = torch.device(model_config["device"])

    loaded_model = AutoModelForCausalLM.from_pretrained(
        model_config["model_path"],
        dtype=model_config["dtype"],
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    assert isinstance(loaded_model, SupportedModel)

    model: Any = loaded_model
    if lora_config["enabled"]:
        model = load_or_build_lora_model(loaded_model, lora_config)
        disable_lora_adapters(model)

    for _, param in model.named_parameters():
        param.requires_grad = False

    if train_config["use_cache"]:
        train_cache.model_cache[model_key] = model
    return model


def load_samples(train_config: TrainConfig, tokenizer: TokenizerType) -> list[TrainSample]:
    samples: list[TrainSample] = []
    for data_config in train_config["data_configs"]:
        eval_generator = get_ret_eval_generator(
            name=data_config["dataset_name"],
            num_samples=data_config["num_samples"],
            num_data_strs=data_config["num_data_strs"],
            num_shots=data_config["num_shots"],
            subset=data_config["subset"],
            split=data_config["split"],
            seed=data_config["seed"],
            data_kwargs=data_config["data_kwargs"],
            template=data_config["template"],
            template_kwargs=data_config["template_kwargs"],
            tokenizer=tokenizer,
        )
        samples.extend(
            TrainSample(
                prompt=compile_prompt(tokenizer, sample["prompt"]),
                semantic_key=generation_cache_key(sample["semantic"]),
            )
            for sample in eval_generator
        )
    return samples


def load_packet_wrapper_for_train(
    train_config: TrainConfig,
    model: Any,
) -> PacketWrapper | None:
    packet_config = train_config["packet_wrapper"]
    if not packet_config["enabled"]:
        return None

    model_device = get_model_device(model)
    wrapper_dtype_name = packet_config["dtype"] or train_config["model"]["dtype"]
    wrapper_dtype = dtype_map[wrapper_dtype_name]

    if packet_config["init_path"] is not None:
        packet_wrapper = load_wrapper(packet_config["init_path"], device=model_device)
        packet_wrapper.to(dtype=wrapper_dtype)
    else:
        body = model.get_base_model() if hasattr(model, "get_base_model") else model
        embed_tokens = body.model.embed_tokens
        mean = torch.mean(embed_tokens.weight).item()
        std = torch.std(embed_tokens.weight).item()
        assert model.config.hidden_size is not None, "Model config must have hidden_size defined."
        assert packet_config["header_len"] is not None
        assert packet_config["trailer_len"] is not None
        packet_wrapper = PacketWrapper(
            header_len=packet_config["header_len"],
            trailer_len=packet_config["trailer_len"],
            dim=model.config.hidden_size,
            dtype=wrapper_dtype,
            mean=mean,
            std=std,
            device=model_device,
        )

    assert packet_wrapper.dim == model.config.hidden_size, (
        f"Packet wrapper dim {packet_wrapper.dim} does not match model hidden size {model.config.hidden_size}"
    )
    return packet_wrapper


def prepare_generation_cache(
    train_config: TrainConfig,
    samples: list[TrainSample],
    model: Any,
    tokenizer: TokenizerType,
    generation_config: GenerationConfig | None,
    debug_recorder: DebugRecorder | None = None,
) -> GenerationCacheAccess:
    cache_device = torch.device(train_config["cache_device"])
    cache_paths = train_config["cache_path"]
    store_logits = train_config["loss"]["type"] == "kl"

    if cache_paths is not None:
        missing_paths = [path for path in cache_paths if not os.path.isdir(path)]
        if missing_paths:
            raise FileNotFoundError(
                f"Configured generation cache artifact not found: {missing_paths[0]}"
            )
        generation_cache = load_generation_cache(
            cache_paths,
            cache_device=cache_device,
        )
        expected_model_path = train_config["model"]["model_path"]
        expected_provenance = {
            "model_path": expected_model_path,
            "tokenizer_path": expected_model_path,
            "dtype": train_config["model"]["dtype"],
        }
        incompatible_fields = [
            field for field, expected in expected_provenance.items()
            if generation_cache.provenance[field] != expected
        ]
        current_tokenizer_state = {
            "padding_side": getattr(tokenizer, "padding_side", None),
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        }
        if generation_cache.provenance["tokenizer"] != current_tokenizer_state:
            incompatible_fields.append("tokenizer")
        if incompatible_fields:
            raise ValueError(
                "Generation cache is incompatible with the configured training model: "
                + ", ".join(incompatible_fields)
            )
        LOGGER.info(
            "Loaded generation cache from %s, size: %d",
            cache_paths,
            len(generation_cache),
        )
        missing_keys = {
            sample["semantic_key"]
            for sample in samples
            if sample["semantic_key"] not in generation_cache
        }
        if missing_keys:
            raise KeyError(
                f"Generation cache is missing {len(missing_keys)} required semantic samples."
            )
        if store_logits:
            missing_logits: set[str] = set()
            for sample in samples:
                key = sample["semantic_key"]
                metadata = generation_cache.metadata(key)
                assert metadata is not None
                if len(metadata.logits) != metadata.num_sequences:
                    missing_logits.add(key)
            if missing_logits:
                raise ValueError(
                    f"Generation cache has {len(missing_logits)} semantic samples without KL logits."
                )
        return generation_cache

    generation_cache = GenerationCache(device=cache_device)
    lora_context = lora_adapters_disabled(model) if train_config["lora"]["enabled"] else nullcontext()
    with lora_context:
        model.eval()
        generation_cache, _ = build_generation_cache(
            samples=samples,
            batch_size=train_config["gen_batch_size"],
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            generation_cache=generation_cache,
            store_logits=store_logits,
            debug_recorder=debug_recorder,
        )

    return generation_cache


def build_optimizers(
    train_config: TrainConfig,
    model: Any,
    packet_wrapper: PacketWrapper | None,
    num_samples: int,
) -> tuple[
    dict[TrainTarget, torch.optim.Optimizer],
    dict[TrainTarget, torch.optim.lr_scheduler.LRScheduler],
]:
    optimizers: dict[TrainTarget, torch.optim.Optimizer] = {}
    schedulers: dict[TrainTarget, torch.optim.lr_scheduler.LRScheduler] = {}
    iter_per_epoch = max(1, (num_samples + train_config["batch_size"] - 1) // train_config["batch_size"])
    total_iter = iter_per_epoch * train_config["total_epoch"]

    for target in train_config["train"]["targets"]:
        params = target_parameters(model, packet_wrapper, target)
        if not params:
            raise ValueError(f"No trainable parameters found for target {target}.")
        optimizer_config = train_config["optimizers"][target]
        optimizer = torch.optim.AdamW(params=params, **optimizer_config["opt_config"])
        scheduler_config = optimizer_config["scheduler_config"].copy()
        if scheduler_config.get("total_iters", 0) == 0:
            scheduler_config["total_iters"] = total_iter
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, **scheduler_config)
        optimizers[target] = optimizer
        schedulers[target] = scheduler

    return optimizers, schedulers


def save_trained_targets(
    train_config: TrainConfig,
    model: Any,
    packet_wrapper: PacketWrapper | None,
    debug_recorder: DebugRecorder | None = None,
) -> None:
    output_dir = train_config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    if "lora" in train_config["train"]["targets"]:
        adapter_name = train_config["lora"]["adapter_name"]
        lora_path = os.path.join(output_dir, "lora")
        save_lora_adapter(model, lora_path, adapter_name)
        LOGGER.info(
            "Saved LoRA adapter to %s",
            lora_path,
        )
        if debug_recorder is not None and debug_recorder.enabled:
            debug_recorder.record_json(
                "final_lora_adapter",
                {
                    "path": lora_path,
                    "adapter_name": adapter_name,
                },
            )

    if "packet_wrapper" in train_config["train"]["targets"]:
        assert packet_wrapper is not None
        state_dict = packet_wrapper.state_dict()
        state_dict["train_config"] = train_config  # type: ignore[typeddict-item]
        wrapper_path = os.path.join(output_dir, "packet_wrapper.pt")
        torch.save(WrapperStateDict(**state_dict), wrapper_path)
        LOGGER.info("Saved PacketWrapper to %s", wrapper_path)
        if debug_recorder is not None and debug_recorder.enabled:
            debug_recorder.record_json(
                "final_packet_wrapper",
                {
                    "path": wrapper_path,
                    "header_len": packet_wrapper.header_len,
                    "trailer_len": packet_wrapper.trailer_len,
                },
            )


def allocate_train_run_dir(
    method_root: str | Path,
    run_suffix: str | None,
    now: datetime | None = None,
) -> Path:
    return allocate_run_dir(method_root, run_suffix, now=now)


def select_resume_run_dir(method_root: str | Path, resume_from: str | Path) -> Path:
    try:
        return select_existing_run_dir(method_root, resume_from)
    except ValueError as exc:
        raise ValueError(
            "--resume-from must name a concrete timestamp run directory that is a "
            "direct child of the configured output_dir."
        ) from exc


def train_one_config(
    train_config: TrainConfig,
    train_cache: TrainCache,
    debug_recorder: DebugRecorder | None = None,
    resume: bool = False,
) -> None:
    LOGGER.info(
        "Training configuration:\n%s",
        pformat(_format_log_floats(train_config)),
    )

    tokenizer = load_tokenizer(train_config, train_cache)
    model = load_model(train_config, train_cache)
    packet_wrapper = load_packet_wrapper_for_train(train_config, model)
    configure_trainable_parameters(model, packet_wrapper, train_config["train"]["targets"])

    samples = load_samples(train_config, tokenizer)
    num_samples = len(samples)
    LOGGER.info("Total training samples: %d", num_samples)
    if num_samples == 0:
        raise ValueError("No training samples loaded.")
    if num_samples % train_config["batch_size"] != 0:
        LOGGER.warning(
            "Number of samples %d is not divisible by batch size %d.",
            num_samples,
            train_config["batch_size"],
        )

    generation_config: GenerationConfig | None = None
    if train_config["cache_path"] is None:
        if model.generation_config is None:
            raise ValueError("Online teacher generation requires a model generation config.")
        generation_kwargs = train_config["model"]["generation_kwargs"].copy()
        generation_kwargs.setdefault("pad_token_id", tokenizer.pad_token_id)
        generation_kwargs.setdefault("do_sample", False)
        generation_config = GenerationConfig(**generation_kwargs)
    generation_cache = prepare_generation_cache(
        train_config=train_config,
        samples=samples,
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
        debug_recorder=debug_recorder,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    saved_state: dict[str, Any] | None = None
    resume_path = checkpoint_path(train_config)
    if resume:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Training checkpoint not found: {resume_path}")
        saved_state = load_training_checkpoint(
            resume_path,
            model=model,
            packet_wrapper=packet_wrapper,
        )

    optimizers, schedulers = build_optimizers(train_config, model, packet_wrapper, num_samples)
    if saved_state is None:
        random.seed(train_config["seed"])
        start_epoch = 0
        epoch_indices = list(range(num_samples))
    else:
        start_epoch, epoch_indices = restore_training_state(
            saved_state,
            optimizers=optimizers,
            schedulers=schedulers,
        )
        LOGGER.info("Resumed training from %s at epoch %d", resume_path, start_epoch + 1)

    attention_runtime = (
        create_train_attention_runtime(
            model,
            train_config["forward_batch_size"],
            train_config["attention_backend"],
        )
        if start_epoch < train_config["total_epoch"]
        else None
    )

    for epoch in range(start_epoch, train_config["total_epoch"]):
        assert attention_runtime is not None
        LOGGER.info("Epoch %d/%d", epoch + 1, train_config["total_epoch"])
        random.shuffle(epoch_indices)
        generation_cache = train_components(
            samples=samples,
            model=model,
            batch_size=train_config["batch_size"],
            forward_batch_size=train_config["forward_batch_size"],
            optimizers=optimizers,
            schedulers=schedulers,
            generation_cache=generation_cache,
            loss_config=train_config["loss"],
            lora_enabled=train_config["lora"]["enabled"],
            lora_adapter_name=(
                train_config["lora"]["adapter_name"]
                if train_config["lora"]["enabled"]
                else None
            ),
            packet_wrapper=packet_wrapper,
            kv_gradient_checkpointing=train_config["kv_gradient_checkpointing"],
            epoch=epoch,
            total_epochs=train_config["total_epoch"],
            epoch_indices=epoch_indices,
            debug_recorder=debug_recorder,
            attention_runtime=attention_runtime,
        )
        save_training_checkpoint(
            resume_path,
            next_epoch=epoch + 1,
            epoch_indices=epoch_indices,
            model=model,
            packet_wrapper=packet_wrapper,
            optimizers=optimizers,
            schedulers=schedulers,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_trained_targets(
        train_config=train_config,
        model=model,
        packet_wrapper=packet_wrapper,
        debug_recorder=debug_recorder,
    )


def run_one_config(
    config_file: str,
    train_cache: TrainCache,
    runtime_overrides: RuntimeCLIOverrides | None = None,
    cli_args: dict[str, object] | None = None,
    config_index: int | None = None,
    config_count: int | None = None,
    discovered_configs: list[str] | None = None,
    resume_from: str | None = None,
    run_suffix_override: str | None = None,
    attention_backend_override: str | None = None,
) -> None:
    train_config_json = load_config_file(
        config_file,
        default_config_file="_default.json",
    )
    train_config_json = apply_runtime_overrides(train_config_json, runtime_overrides)
    if attention_backend_override is not None:
        train_config_json["attention_backend"] = attention_backend_override
    train_config = load_train_config(train_config_json)
    method_root = train_config["output_dir"]
    if resume_from is None:
        effective_suffix = (
            run_suffix_override
            if run_suffix_override is not None
            else train_config["run_suffix"]
        )
        run_dir = allocate_train_run_dir(method_root, effective_suffix)
    else:
        effective_suffix = train_config["run_suffix"]
        run_dir = select_resume_run_dir(method_root, resume_from)

    run_config = copy.deepcopy(train_config)
    run_config["output_dir"] = str(run_dir)
    run_config["run_suffix"] = effective_suffix

    with RuntimeContext(
        entrypoint="run_train",
        run_dir=run_dir,
        config_file=config_file,
        resolved_config=run_config,
        config_snapshot_name="train_config.json",
        cli_args={} if cli_args is None else cli_args,
    ) as runtime_context:
        if config_index == 0 and config_count is not None:
            runtime_context.logger.info("Loaded %d training configurations.", config_count)
            for discovered_config in discovered_configs or []:
                runtime_context.logger.info(" - %s", discovered_config)
        train_one_config(
            run_config,
            train_cache,
            debug_recorder=runtime_context.debug_recorder,
            resume=resume_from is not None,
        )
        resume_path = checkpoint_path(run_config)
        if os.path.exists(resume_path):
            os.unlink(resume_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_files_or_paths",
        type=str,
        nargs="+",
        help="Path to training configuration files or directories.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume one training configuration from a concrete timestamp run directory.",
    )
    parser.add_argument(
        "--run-suffix",
        type=str,
        default=None,
        help="Override the optional readable suffix for a new timestamp run directory.",
    )
    parser.add_argument(
        "--attention-backend",
        choices=("auto", "flex", "sdpa"),
        default=None,
        help="Override the training attention backend preference.",
    )
    add_runtime_cli_args(parser)
    args = parser.parse_args()
    runtime_overrides = runtime_overrides_from_args(args)
    cli_args = vars(args).copy()

    all_config_files: list[str] = []
    for file_or_path in args.config_files_or_paths:
        all_config_files.extend(gather_config_files(file_or_path, pattern=r".*\.json$"))

    if not all_config_files:
        logging.basicConfig(level=logging.INFO)
        LOGGER.info("Loaded 0 training configurations.")
        sys.exit(0)

    if args.resume_from is not None and len(all_config_files) != 1:
        parser.error("--resume-from requires exactly one training configuration.")
    if args.resume_from is not None and args.run_suffix is not None:
        parser.error("--run-suffix cannot be used with --resume-from.")

    train_cache = TrainCache()
    for config_index, config_file in enumerate(all_config_files):
        run_one_config(
            config_file,
            train_cache,
            runtime_overrides=runtime_overrides,
            cli_args=cli_args,
            config_index=config_index,
            config_count=len(all_config_files),
            discovered_configs=all_config_files,
            resume_from=args.resume_from,
            run_suffix_override=args.run_suffix,
            attention_backend_override=args.attention_backend,
        )
