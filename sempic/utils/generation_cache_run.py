"""Standalone generation-cache configuration and artifact publication."""

import copy
import ctypes
import errno
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any, TypedDict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from sempic.dataset import get_ret_eval_generator
from sempic.prompt import compile_prompt
from sempic.utils.generate import (
    TokenizerType,
    generation_cache_key,
    resolve_generation_config,
)
from sempic.utils.generation_cache import (
    StreamingGenerationCacheWriter,
    generation_cache_provenance,
)
from sempic.utils.run_storage import atomic_write_json
from sempic.utils.train import (
    DatasetConfig,
    TrainSample,
    build_generation_cache,
    dtype_map,
)

_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


class GenerationCacheModelConfig(TypedDict):
    model_path: str
    tokenizer_path: str
    dtype: str
    device: str
    generation_kwargs: dict[str, Any]


class GenerationCacheConfig(TypedDict):
    model: GenerationCacheModelConfig
    data_configs: list[DatasetConfig]
    store_logits: bool
    gen_batch_size: int
    cache_device: str
    seed: int
    output_dir: str


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOTSUP, "Atomic no-replace directory rename is unavailable.")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(
                error_number,
                os.strerror(error_number),
                destination,
            )
        raise OSError(error_number, os.strerror(error_number), destination)


def _require_non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string.")
    return value


def _require_integer(value: Any, field: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be at least {minimum}.")
    return value


def _require_dict(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object.")
    return copy.deepcopy(value)


def _load_dataset_config(value: Any, index: int) -> DatasetConfig:
    field = f"data_configs[{index}]"
    data = _require_dict(value, field)
    return DatasetConfig(
        dataset_name=_require_non_empty_string(
            data.get("dataset_name"), f"{field}.dataset_name"
        ),
        num_samples=_require_integer(
            data.get("num_samples"), f"{field}.num_samples", minimum=0
        ),
        num_data_strs=_require_integer(
            data.get("num_data_strs"), f"{field}.num_data_strs", minimum=0
        ),
        num_shots=_require_integer(
            data.get("num_shots"), f"{field}.num_shots", minimum=0
        ),
        subset=_require_non_empty_string(data.get("subset"), f"{field}.subset"),
        split=_require_non_empty_string(data.get("split", "train"), f"{field}.split"),
        seed=_require_integer(data.get("seed", 42), f"{field}.seed"),
        data_kwargs=_require_dict(data.get("data_kwargs", {}), f"{field}.data_kwargs"),
        template=_require_non_empty_string(
            data.get("template", "default"), f"{field}.template"
        ),
        template_kwargs=_require_dict(
            data.get("template_kwargs", {}), f"{field}.template_kwargs"
        ),
    )


def load_generation_cache_config(config: dict[str, Any]) -> GenerationCacheConfig:
    if not isinstance(config, dict):
        raise ValueError("Generation-cache config must be an object.")
    model_data = _require_dict(config.get("model"), "model")
    model_path = _require_non_empty_string(model_data.get("model_path"), "model.model_path")
    dtype = _require_non_empty_string(model_data.get("dtype", "bfloat16"), "model.dtype")
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported model.dtype: {dtype}")
    device = _require_non_empty_string(model_data.get("device", "cuda:0"), "model.device")
    if device != "auto":
        try:
            torch.device(device)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"Invalid model.device: {device}") from exc

    raw_data_configs = config.get("data_configs")
    if not isinstance(raw_data_configs, list) or not raw_data_configs:
        raise ValueError("data_configs must be a non-empty list.")
    data_configs = [
        _load_dataset_config(value, index)
        for index, value in enumerate(raw_data_configs)
    ]

    store_logits = config.get("store_logits")
    if not isinstance(store_logits, bool):
        raise ValueError("store_logits must be a boolean.")
    cache_device = _require_non_empty_string(
        config.get("cache_device", "cpu"), "cache_device"
    )
    try:
        parsed_cache_device = torch.device(cache_device)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid cache_device: {cache_device}") from exc
    if parsed_cache_device.type != "cpu":
        raise ValueError("Generation-cache streaming requires cache_device to be CPU.")

    return GenerationCacheConfig(
        model=GenerationCacheModelConfig(
            model_path=model_path,
            tokenizer_path=_require_non_empty_string(
                model_data.get("tokenizer_path", model_path), "model.tokenizer_path"
            ),
            dtype=dtype,
            device=device,
            generation_kwargs=_require_dict(
                model_data.get("generation_kwargs", {}), "model.generation_kwargs"
            ),
        ),
        data_configs=data_configs,
        store_logits=store_logits,
        gen_batch_size=_require_integer(
            config.get("gen_batch_size", 1), "gen_batch_size", minimum=1
        ),
        cache_device=cache_device,
        seed=_require_integer(config.get("seed", 42), "seed"),
        output_dir=_require_non_empty_string(config.get("output_dir"), "output_dir"),
    )


def _load_resources(config: GenerationCacheConfig) -> tuple[Any, TokenizerType]:
    model_config = config["model"]
    tokenizer: TokenizerType = AutoTokenizer.from_pretrained(
        model_config["tokenizer_path"]
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define an EOS token when no pad token is set.")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device_map: str | torch.device = (
        "auto"
        if model_config["device"] == "auto"
        else torch.device(model_config["device"])
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_config["model_path"],
        dtype=dtype_map[model_config["dtype"]],
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    if model.generation_config is None:
        raise ValueError("Teacher model must define a generation config.")
    model.eval()
    return model, tokenizer


def _load_samples(
    config: GenerationCacheConfig,
    tokenizer: TokenizerType,
) -> list[TrainSample]:
    samples: list[TrainSample] = []
    for data_config in config["data_configs"]:
        entries = get_ret_eval_generator(
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
                prompt=compile_prompt(tokenizer, entry["prompt"]),
                semantic_key=generation_cache_key(entry["semantic"]),
            )
            for entry in entries
        )
    if not samples:
        raise ValueError("No generation-cache samples loaded.")
    return samples


def _resolved_config(
    config: GenerationCacheConfig,
    *,
    output_dir: Path,
    tokenizer: TokenizerType,
    generation_config: GenerationConfig,
) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    resolved["output_dir"] = str(output_dir)
    resolved["cache_path"] = str(output_dir)
    resolved["model"]["generation_kwargs"] = generation_config.to_dict()
    resolved["model"]["tokenizer"] = {
        "padding_side": tokenizer.padding_side,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    return resolved


def generate_cache_artifact(config: GenerationCacheConfig) -> Path:
    output_dir = Path(config["output_dir"]).expanduser().absolute()
    if os.path.lexists(output_dir):
        raise FileExistsError(f"Generation-cache output target already exists: {output_dir}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(output_dir):
        raise FileExistsError(f"Generation-cache output target already exists: {output_dir}")
    temporary_dir = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.",
        suffix=".tmp",
        dir=output_dir.parent,
    ))
    try:
        random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        model, tokenizer = _load_resources(config)
        samples = _load_samples(config, tokenizer)

        generation_kwargs = copy.deepcopy(config["model"]["generation_kwargs"])
        generation_kwargs.setdefault("pad_token_id", tokenizer.pad_token_id)
        generation_kwargs.setdefault("do_sample", False)
        requested_generation = GenerationConfig(**generation_kwargs)
        effective_generation = resolve_generation_config(model, requested_generation)
        resolved_config = _resolved_config(
            config,
            output_dir=output_dir,
            tokenizer=tokenizer,
            generation_config=effective_generation,
        )
        with StreamingGenerationCacheWriter(
            temporary_dir,
            provenance=generation_cache_provenance(resolved_config),
        ) as writer:
            build_generation_cache(
                samples=samples,
                batch_size=config["gen_batch_size"],
                model=model,
                tokenizer=tokenizer,
                generation_config=effective_generation,
                store_logits=config["store_logits"],
                generation_sink=writer.add,
            )
            writer.finalize()
        atomic_write_json(
            temporary_dir / "resolved_config.json",
            resolved_config,
        )
        if os.path.lexists(output_dir):
            raise FileExistsError(
                f"Generation-cache output target already exists: {output_dir}"
            )
        _rename_directory_noreplace(temporary_dir, output_dir)
        return output_dir
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
