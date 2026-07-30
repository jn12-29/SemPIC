import argparse
import copy
import json
import logging
import re
from contextlib import contextmanager
from contextvars import ContextVar
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Iterator, TypedDict

import torch

from .run_storage import atomic_write_json


class LoggingConfig(TypedDict):
    level: str


class DebugDumpConfig(TypedDict):
    enabled: bool
    sample_limit: int
    save_token_ids: bool
    save_tensor_values: bool


class RuntimeCLIOverrides(TypedDict, total=False):
    debug_enabled: bool | None
    log_level: str | None
    debug_sample_limit: int | None


def load_logging_config(config: dict[str, Any]) -> LoggingConfig:
    logging_config = config.get("logging", {})
    return LoggingConfig(
        level=logging_config.get("level", "INFO"),
    )


def load_debug_dump_config(config: dict[str, Any]) -> DebugDumpConfig:
    debug_config = config.get("debug_dump", {})
    return DebugDumpConfig(
        enabled=debug_config.get("enabled", False),
        sample_limit=debug_config.get("sample_limit", 2),
        save_token_ids=debug_config.get("save_token_ids", False),
        save_tensor_values=debug_config.get("save_tensor_values", False),
    )


def add_runtime_cli_args(parser: argparse.ArgumentParser) -> None:
    debug_group = parser.add_mutually_exclusive_group()
    debug_group.add_argument(
        "--debug",
        dest="debug_enabled",
        action="store_true",
        default=None,
        help="Enable debug artifact dumps.",
    )
    debug_group.add_argument(
        "--no-debug",
        dest="debug_enabled",
        action="store_false",
        help="Disable debug artifact dumps.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Override the logging level.",
    )
    parser.add_argument(
        "--debug-sample-limit",
        type=int,
        default=None,
        help="Override the number of samples that write debug artifacts.",
    )


def runtime_overrides_from_args(args: argparse.Namespace) -> RuntimeCLIOverrides:
    return RuntimeCLIOverrides(
        debug_enabled=getattr(args, "debug_enabled", None),
        log_level=getattr(args, "log_level", None),
        debug_sample_limit=getattr(args, "debug_sample_limit", None),
    )


def apply_runtime_overrides(
    config: dict[str, Any],
    overrides: RuntimeCLIOverrides | None = None,
) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    resolved["logging"] = load_logging_config(resolved)
    resolved["debug_dump"] = load_debug_dump_config(resolved)
    if overrides is None:
        return resolved

    debug_enabled = overrides.get("debug_enabled", None)
    if debug_enabled is not None:
        resolved["debug_dump"]["enabled"] = debug_enabled
    log_level = overrides.get("log_level", None)
    if log_level is not None:
        resolved["logging"]["level"] = log_level
    debug_sample_limit = overrides.get("debug_sample_limit", None)
    if debug_sample_limit is not None:
        resolved["debug_dump"]["sample_limit"] = debug_sample_limit
    return resolved


def _safe_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return safe or "run"


def _jsonable(value: Any, save_tensor_values: bool = False) -> Any:
    if isinstance(value, torch.Tensor):
        summary: dict[str, Any] = tensor_summary(value)
        if save_tensor_values:
            summary["values"] = value.detach().cpu().tolist()
        return summary
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item, save_tensor_values=save_tensor_values)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _jsonable(item, save_tensor_values=save_tensor_values)
            for item in value
        ]
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    return value


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "requires_grad": tensor.requires_grad,
    }


def position_range_summary(position_ids: torch.Tensor) -> dict[str, Any]:
    summary = tensor_summary(position_ids)
    if position_ids.numel() > 0:
        flat = position_ids.reshape(-1)
        summary["first"] = int(flat[0].detach().cpu().item())
        summary["last"] = int(flat[-1].detach().cpu().item())
    return summary


def kv_cache_summary(kv_cache: Any) -> dict[str, Any]:
    raw_cache = getattr(kv_cache, "_cache", None)
    if not isinstance(raw_cache, dict):
        return {"type": type(kv_cache).__name__}

    layers: dict[str, Any] = {}
    for layer in sorted(raw_cache.keys()):
        chunks = []
        for key_value in raw_cache[layer]:
            key = getattr(key_value, "key", None)
            value = getattr(key_value, "value", None)
            position_ids = getattr(key_value, "position_ids", None)
            chunk: dict[str, Any] = {}
            if isinstance(key, torch.Tensor):
                chunk["key"] = tensor_summary(key)
            if isinstance(value, torch.Tensor):
                chunk["value"] = tensor_summary(value)
            if isinstance(position_ids, torch.Tensor):
                chunk["position_ids"] = position_range_summary(position_ids)
            chunks.append(chunk)
        layers[str(layer)] = chunks
    return {
        "type": type(kv_cache).__name__,
        "num_layers": len(layers),
        "layers": layers,
    }


def kv_caches_summary(kv_caches: list[Any]) -> list[dict[str, Any]]:
    return [kv_cache_summary(kv_cache) for kv_cache in kv_caches]


def hf_cache_summary(cache: Any) -> dict[str, Any]:
    layers_obj = getattr(cache, "layers", None)
    if layers_obj is None:
        return {"type": type(cache).__name__}

    layers = []
    for layer in layers_obj:
        layer_summary: dict[str, Any] = {}
        key = getattr(layer, "keys", None)
        value = getattr(layer, "values", None)
        if isinstance(key, torch.Tensor):
            layer_summary["key"] = tensor_summary(key)
        if isinstance(value, torch.Tensor):
            layer_summary["value"] = tensor_summary(value)
        layers.append(layer_summary)
    return {
        "type": type(cache).__name__,
        "num_layers": len(layers),
        "layers": layers,
    }


def generation_summary(
    generation: dict[str, Any],
    save_token_ids: bool = False,
) -> dict[str, Any]:
    sequences = generation.get("sequences", [])
    logits = generation.get("logits", [])
    texts = generation.get("text", [])
    summary: dict[str, Any] = {
        "num_sequences": len(sequences),
        "sequence_lengths": [
            int(sequence.numel()) if isinstance(sequence, torch.Tensor) else None
            for sequence in sequences
        ],
        "logit_shapes": [
            list(logit.shape) if isinstance(logit, torch.Tensor) else None
            for logit in logits
        ],
        "texts": [
            {"text": text, "length": len(text)}
            for text in texts
            if isinstance(text, str)
        ],
    }
    if save_token_ids:
        summary["sequence_token_ids"] = [
            sequence.detach().cpu().tolist()
            for sequence in sequences
            if isinstance(sequence, torch.Tensor)
        ]
    return summary


class DebugRecorder:
    def __init__(
        self,
        debug_dir: str | Path,
        config: DebugDumpConfig,
        enabled: bool | None = None,
    ):
        self.debug_dir = Path(debug_dir)
        self.config = config
        self.enabled = config["enabled"] if enabled is None else enabled
        self._sample_count = 0

    def child(self, name: str) -> "DebugRecorder":
        return DebugRecorder(
            self.debug_dir / _safe_name(name),
            self.config,
            enabled=self.enabled,
        )

    def sample_scope(self, name: str, index: int | None = None) -> "DebugRecorder":
        if not self.enabled or self._sample_count >= self.config["sample_limit"]:
            return DebugRecorder(
                self.debug_dir / _safe_name(name),
                self.config,
                enabled=False,
            )
        self._sample_count += 1
        prefix = f"{index:04d}_" if index is not None else ""
        return self.child(prefix + name)

    def record_json(self, name: str, data: Any) -> None:
        if not self.enabled:
            return
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        path = self.debug_dir / f"{_safe_name(name)}.json"
        with open(path, "w") as f:
            json.dump(
                _jsonable(
                    data,
                    save_tensor_values=self.config["save_tensor_values"],
                ),
                f,
                indent=2,
            )

    def record_text(self, name: str, text: str) -> None:
        if not self.enabled:
            return
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        path = self.debug_dir / f"{_safe_name(name)}.txt"
        with open(path, "w") as f:
            f.write(text)

    def token_ids(self, tensor: torch.Tensor) -> Any:
        if self.config["save_token_ids"]:
            return tensor.detach().cpu().tolist()
        return tensor_summary(tensor)


_CURRENT_DEBUG_RECORDER: ContextVar[DebugRecorder | None] = ContextVar(
    "CURRENT_DEBUG_RECORDER",
    default=None,
)


@contextmanager
def debug_recording_scope(recorder: DebugRecorder | None) -> Iterator[None]:
    token = _CURRENT_DEBUG_RECORDER.set(recorder)
    try:
        yield
    finally:
        _CURRENT_DEBUG_RECORDER.reset(token)


def get_current_debug_recorder() -> DebugRecorder | None:
    return _CURRENT_DEBUG_RECORDER.get()


class RuntimeContext(AbstractContextManager["RuntimeContext"]):
    def __init__(
        self,
        entrypoint: str,
        run_dir: str | Path,
        config_file: str | None,
        resolved_config: dict[str, Any] | None,
        config_snapshot_name: str | None,
        cli_args: dict[str, Any],
    ):
        snapshot_fields = (config_file, resolved_config, config_snapshot_name)
        if not (all(value is None for value in snapshot_fields) or all(
            value is not None for value in snapshot_fields
        )):
            raise ValueError(
                "config_file, resolved_config, and config_snapshot_name must be all set "
                "or all None."
            )
        self.entrypoint = entrypoint
        self.config_file = config_file
        self.resolved_config = resolved_config
        self.config_snapshot_name = config_snapshot_name
        self.cli_args = cli_args
        self.run_dir = Path(run_dir)
        self.debug_dir = self.run_dir / "debug"
        self.logger = logging.getLogger(f"sempic.{entrypoint}.{id(self)}")
        self.package_logger = logging.getLogger("sempic")
        self._handlers: list[logging.Handler] = []
        self._previous_package_level: int | None = None
        self._previous_package_propagate: bool | None = None
        self._previous_logger_level: int | None = None
        self._previous_logger_propagate: bool | None = None
        self.debug_recorder = DebugRecorder(
            self.debug_dir,
            (
                resolved_config["debug_dump"]
                if resolved_config is not None
                else load_debug_dump_config({})
            ),
        )

    def _restore_logging(self) -> None:
        for handler in self._handlers:
            self.package_logger.removeHandler(handler)
            handler.close()
        self._handlers = []
        if self._previous_package_level is not None:
            self.package_logger.setLevel(self._previous_package_level)
        if self._previous_package_propagate is not None:
            self.package_logger.propagate = self._previous_package_propagate
        if self._previous_logger_level is not None:
            self.logger.setLevel(self._previous_logger_level)
        if self._previous_logger_propagate is not None:
            self.logger.propagate = self._previous_logger_propagate

    def __enter__(self) -> "RuntimeContext":
        if not self.run_dir.is_dir():
            raise ValueError(f"Run directory is not a directory: {self.run_dir}")
        logging_config = (
            self.resolved_config["logging"]
            if self.resolved_config is not None
            else load_logging_config({})
        )
        level_name = (
            self.cli_args.get("log_level") or logging_config["level"]
        ).upper()
        level = getattr(logging, level_name, logging.INFO)
        self._previous_package_level = self.package_logger.level
        self._previous_package_propagate = self.package_logger.propagate
        self._previous_logger_level = self.logger.level
        self._previous_logger_propagate = self.logger.propagate
        self.package_logger.setLevel(level)
        self.package_logger.propagate = False
        self.logger.setLevel(level)
        self.logger.propagate = True
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        file_handler = logging.FileHandler(self.run_dir / "run.log")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self._handlers = [file_handler, console_handler]
        for handler in self._handlers:
            self.package_logger.addHandler(handler)
        try:
            if self.resolved_config is not None:
                assert self.config_snapshot_name is not None
                self.write_json("resolved_config.json", self.resolved_config)
                self.write_json(self.config_snapshot_name, self.resolved_config)
            self.write_json("cli_args.json", self.cli_args)
            if self.config_file is None:
                self.logger.info("Started %s invocation", self.entrypoint)
            else:
                self.logger.info("Started %s for %s", self.entrypoint, self.config_file)
            self.logger.info("Run directory: %s", self.run_dir)
        except BaseException:
            self._restore_logging()
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_type is None:
            if self.config_file is None:
                self.logger.info("Completed %s invocation", self.entrypoint)
            else:
                self.logger.info("Completed %s for %s", self.entrypoint, self.config_file)
        else:
            if self.config_file is None:
                self.logger.exception("Failed %s invocation", self.entrypoint)
            else:
                self.logger.exception("Failed %s for %s", self.entrypoint, self.config_file)
        self._restore_logging()
        return False

    def write_json(self, file_name: str, data: Any) -> None:
        atomic_write_json(self.run_dir / file_name, _jsonable(data))
