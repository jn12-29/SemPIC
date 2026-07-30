"""Configuration and resource bootstrap shared by evaluation consumers."""

import gc
import hashlib
import logging
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterator, TypedDict

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from ..cache.compress import PRESS_CLASSES, ScorerPress
from ..cache_comb import get_cache_comb_func
from ..dataset import get_ret_eval_generator
from ..dataset.abc import RetEvalEntry
from ..model import SupportedModel
from ..packet_wrapper import PacketWrapper, load_wrapper
from ..utils.lora import disable_lora_adapters
from ..utils.runtime import (
    DebugDumpConfig,
    LoggingConfig,
    load_debug_dump_config,
    load_logging_config,
)


LORA_EVAL_ADAPTER_NAME = "lora_kv_cache_eval"
LOGGER = logging.getLogger(__name__)
TokenizerType = PreTrainedTokenizer | PreTrainedTokenizerFast
ModelCacheKey = tuple[str, str, str]


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


class CompressConfig(TypedDict):
    method: str
    compression_ratio: float
    keep_filler_tokens: bool
    kwargs: dict


class QuantizationConfig(TypedDict):
    num_bits: int
    axis: int
    group_size: int


class CacheCombConfig(TypedDict):
    method: str
    kwargs: dict


class ArtifactPathConfig(TypedDict):
    path: str | None


class EvalConfig(TypedDict):
    model: ModelConfig
    dataset: DatasetConfig
    cache_comb: CacheCombConfig
    packet_wrapper: ArtifactPathConfig
    lora: ArtifactPathConfig
    compress: CompressConfig | None
    quantization: QuantizationConfig | None
    seed: int
    logging: LoggingConfig
    debug_dump: DebugDumpConfig


class EvalResourceCache(TypedDict):
    model: dict[ModelCacheKey, SupportedModel]
    tokenizer: dict[ModelCacheKey, TokenizerType]
    packet_wrapper: dict[tuple[str, str], PacketWrapper]
    lora_adapter: dict[tuple[ModelCacheKey, str], str]


@dataclass(frozen=True, slots=True)
class EvalResources:
    model: SupportedModel
    tokenizer: TokenizerType
    packet_wrapper: PacketWrapper | None
    lora_adapter_name: str | None


def _validate_artifact_path(field_name: str, artifact_path: str) -> None:
    path_parts = PurePosixPath(artifact_path).parts
    if not artifact_path.startswith("./"):
        raise ValueError(
            f"{field_name} must be a repository-root-relative path starting with './'."
        )
    if PurePosixPath(artifact_path).is_absolute():
        raise ValueError(f"{field_name} must not be an absolute path.")
    if ".." in path_parts:
        raise ValueError(f"{field_name} must not contain a '..' path segment.")
    if "latest" in path_parts:
        raise ValueError(f"{field_name} must not contain a 'latest' path segment.")


def load_eval_config(loaded_json: dict) -> EvalConfig:
    model = ModelConfig(
        model_path=loaded_json["model"]["model_path"],
        dtype=loaded_json["model"].get("dtype", "float32"),
        device=loaded_json["model"].get("device", "cuda:0"),
        generation_kwargs=loaded_json["model"].get("generation_kwargs", {}),
    )

    dataset = DatasetConfig(
        dataset_name=loaded_json["dataset"]["dataset_name"],
        num_samples=loaded_json["dataset"]["num_samples"],
        num_data_strs=loaded_json["dataset"]["num_data_strs"],
        num_shots=loaded_json["dataset"]["num_shots"],
        subset=loaded_json["dataset"]["subset"],
        split=loaded_json["dataset"]["split"],
        seed=loaded_json["dataset"]["seed"],
        data_kwargs=loaded_json["dataset"].get("data_kwargs", {}),
        template=loaded_json["dataset"].get("template", "default"),
        template_kwargs=loaded_json["dataset"].get("template_kwargs", {}),
    )
    cache_comb = CacheCombConfig(
        method=loaded_json["cache_comb"]["method"],
        kwargs=loaded_json["cache_comb"].get("kwargs", {}),
    )
    try:
        get_cache_comb_func(cache_comb["method"])
    except ValueError as exc:
        raise ValueError(f"Unsupported cache_comb.method: {cache_comb['method']}") from exc

    legacy_lora_field = "lora" + "_adapter"
    if legacy_lora_field in loaded_json:
        raise ValueError("Legacy LoRA eval field is no longer supported. Use lora.path.")

    raw_packet_wrapper = loaded_json.get("packet_wrapper", {})
    if raw_packet_wrapper is None:
        raw_packet_wrapper = {}
    if not isinstance(raw_packet_wrapper, dict):
        raise ValueError("packet_wrapper must be an object with a path field.")
    packet_wrapper = ArtifactPathConfig(path=raw_packet_wrapper.get("path", None))
    if packet_wrapper["path"] is not None and (
        not isinstance(packet_wrapper["path"], str) or packet_wrapper["path"] == ""
    ):
        raise ValueError("packet_wrapper.path must be a non-empty string or null.")
    if packet_wrapper["path"] is not None:
        _validate_artifact_path("packet_wrapper.path", packet_wrapper["path"])

    raw_lora = loaded_json.get("lora", {})
    if raw_lora is None:
        raw_lora = {}
    if not isinstance(raw_lora, dict):
        raise ValueError("lora must be an object with a path field.")
    lora = ArtifactPathConfig(path=raw_lora.get("path", None))
    if lora["path"] is not None and (
        not isinstance(lora["path"], str) or lora["path"] == ""
    ):
        raise ValueError("lora.path must be a non-empty string or null.")
    if lora["path"] is not None:
        _validate_artifact_path("lora.path", lora["path"])

    cache_comb_method = cache_comb["method"]
    if cache_comb_method == "kvpacket":
        if packet_wrapper["path"] is None:
            raise ValueError("cache_comb.method 'kvpacket' requires packet_wrapper.path.")
        if lora["path"] is not None:
            raise ValueError(
                "lora.path is only valid with cache_comb.method 'sempic' or 'sempic_kvpacket'."
            )
    elif cache_comb_method == "sempic":
        if lora["path"] is None:
            raise ValueError("cache_comb.method 'sempic' requires lora.path.")
        if packet_wrapper["path"] is not None:
            raise ValueError(
                "packet_wrapper.path is only valid with cache_comb.method 'kvpacket' or 'sempic_kvpacket'."
            )
    elif cache_comb_method == "sempic_kvpacket":
        if packet_wrapper["path"] is None or lora["path"] is None:
            raise ValueError(
                "cache_comb.method 'sempic_kvpacket' requires packet_wrapper.path and lora.path."
            )
    else:
        if packet_wrapper["path"] is not None:
            raise ValueError(
                "packet_wrapper.path is only valid with cache_comb.method 'kvpacket' or 'sempic_kvpacket'."
            )
        if lora["path"] is not None:
            raise ValueError(
                "lora.path is only valid with cache_comb.method 'sempic' or 'sempic_kvpacket'."
            )

    quant_config_dict = loaded_json.get("quantization", None)
    if quant_config_dict is not None:
        quantization_config = QuantizationConfig(
            num_bits=quant_config_dict["num_bits"],
            axis=quant_config_dict.get("axis", 0),
            group_size=quant_config_dict.get("group_size", 64),
        )
    else:
        quantization_config = None

    compress_config_dict = loaded_json.get("compress", None)
    if compress_config_dict is not None:
        compress_config = CompressConfig(
            method=compress_config_dict["method"],
            compression_ratio=compress_config_dict["compression_ratio"],
            keep_filler_tokens=compress_config_dict.get("keep_filler_tokens", False),
            kwargs=compress_config_dict.get("kwargs", {}),
        )
    else:
        compress_config = None

    return EvalConfig(
        model=model,
        dataset=dataset,
        cache_comb=cache_comb,
        packet_wrapper=packet_wrapper,
        lora=lora,
        compress=compress_config,
        quantization=quantization_config,
        seed=loaded_json["seed"],
        logging=load_logging_config(loaded_json),
        debug_dump=load_debug_dump_config(loaded_json),
    )


def load_lora_adapter_for_eval(
    model: SupportedModel,
    adapter_path: str,
    adapter_name: str = LORA_EVAL_ADAPTER_NAME,
) -> None:
    if not hasattr(model, "load_adapter"):
        raise ValueError(
            "This transformers model does not expose load_adapter(); "
            "install a PEFT-compatible transformers/peft version to use LoRA document KV evaluation."
        )
    try:
        model.load_adapter(adapter_path, adapter_name=adapter_name, is_trainable=False)
    except TypeError:
        model.load_adapter(adapter_path, adapter_name=adapter_name)
    if hasattr(model, "set_adapter"):
        model.set_adapter(adapter_name)
    model.to(model.device)
    disable_lora_adapters(model)


def create_eval_resource_cache() -> EvalResourceCache:
    return EvalResourceCache(
        model={}, tokenizer={}, packet_wrapper={}, lora_adapter={}
    )


def release_eval_resource_cache(cache: EvalResourceCache) -> None:
    cache["model"].clear()
    cache["tokenizer"].clear()
    cache["packet_wrapper"].clear()
    cache["lora_adapter"].clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_eval_resources(
    config: EvalConfig,
    cache: EvalResourceCache,
) -> EvalResources:
    packet_wrapper_path = config["packet_wrapper"]["path"]
    lora_path = config["lora"]["path"]
    model_cache_key: ModelCacheKey = (
        config["model"]["model_path"],
        config["model"]["dtype"],
        config["model"]["device"],
    )

    packet_wrapper_key = (
        (packet_wrapper_path, config["model"]["device"])
        if packet_wrapper_path is not None
        else None
    )
    packet_wrapper = (
        cache["packet_wrapper"].get(packet_wrapper_key)
        if packet_wrapper_path is not None
        else None
    )
    model = cache["model"].get(model_cache_key)
    tokenizer = cache["tokenizer"].get(model_cache_key)

    if packet_wrapper is None and packet_wrapper_path is not None:
        packet_wrapper = load_wrapper(
            packet_wrapper_path,
            device=torch.device(config["model"]["device"]),
        )
        LOGGER.info("Packet wrapper loaded %s.", packet_wrapper)
        assert packet_wrapper_key is not None
        cache["packet_wrapper"][packet_wrapper_key] = packet_wrapper

    if model is None or tokenizer is None:
        model = AutoModelForCausalLM.from_pretrained(
            config["model"]["model_path"],
            dtype=config["model"]["dtype"],
            device_map=torch.device(config["model"]["device"]),
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["model_path"])
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id + 2
        assert model.generation_config is not None
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        cache["model"][model_cache_key] = model
        cache["tokenizer"][model_cache_key] = tokenizer

    adapter_name = None
    if lora_path is not None:
        adapter_key = (model_cache_key, lora_path)
        adapter_name = cache["lora_adapter"].get(adapter_key)
        if adapter_name is None:
            digest = hashlib.sha256(lora_path.encode("utf-8")).hexdigest()
            adapter_name = f"{LORA_EVAL_ADAPTER_NAME}_{digest}"
            load_lora_adapter_for_eval(
                model,
                lora_path,
                adapter_name=adapter_name,
            )
            cache["lora_adapter"][adapter_key] = adapter_name
            LOGGER.info("LoRA adapter %s loaded from %s.", adapter_name, lora_path)

    return EvalResources(
        model=model,
        tokenizer=tokenizer,
        packet_wrapper=packet_wrapper,
        lora_adapter_name=adapter_name,
    )


def build_eval_generator(
    config: DatasetConfig,
    tokenizer: TokenizerType,
) -> Iterator[RetEvalEntry]:
    return get_ret_eval_generator(
        name=config["dataset_name"],
        num_samples=config["num_samples"],
        num_data_strs=config["num_data_strs"],
        num_shots=config["num_shots"],
        subset=config["subset"],
        split=config["split"],
        seed=config["seed"],
        data_kwargs=config["data_kwargs"],
        template=config["template"],
        template_kwargs=config["template_kwargs"],
        tokenizer=tokenizer,
    )


def build_generation_config(config: ModelConfig) -> GenerationConfig | None:
    if config["generation_kwargs"]:
        return GenerationConfig(**config["generation_kwargs"])
    return None


def build_compressor(config: CompressConfig | None) -> tuple[ScorerPress | None, bool]:
    if config is None:
        return None, False

    compress_cls = PRESS_CLASSES.get(config["method"])
    if compress_cls is None:
        raise ValueError(f"Unknown compression method: {config['method']}")
    compressor = compress_cls(
        compression_ratio=config["compression_ratio"],
        **config.get("kwargs", {}),
    )
    return compressor, config["keep_filler_tokens"]


__all__ = [
    "LORA_EVAL_ADAPTER_NAME",
    "ArtifactPathConfig",
    "CacheCombConfig",
    "CompressConfig",
    "DatasetConfig",
    "EvalConfig",
    "EvalResourceCache",
    "EvalResources",
    "ModelCacheKey",
    "ModelConfig",
    "QuantizationConfig",
    "build_compressor",
    "build_eval_generator",
    "build_generation_config",
    "create_eval_resource_cache",
    "load_eval_config",
    "load_eval_resources",
    "load_lora_adapter_for_eval",
    "release_eval_resource_cache",
]
