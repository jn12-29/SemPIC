"""Load evaluation configs accepted by attention analysis."""

from __future__ import annotations

from collections.abc import Sequence

from ..evaluation import load_eval_config
from ..utils.config import load_config_file


SUPPORTED_METHODS = (
    "full_recompute",
    "no_recompute",
    "kvpacket",
    "sempic",
    "sempic_kvpacket",
)
LoadedConfig = tuple[str, dict]


def load_analysis_configs(config_files: Sequence[str]) -> list[LoadedConfig]:
    loaded = [
        (
            config_file,
            load_eval_config(
                load_config_file(config_file, default_config_file="_default.json")
            ),
        )
        for config_file in config_files
    ]
    unsupported = {
        config["cache_comb"]["method"]
        for _, config in loaded
        if config["cache_comb"]["method"] not in SUPPORTED_METHODS
    }
    if unsupported:
        raise ValueError(
            f"Attention analysis does not support methods: {sorted(unsupported)}."
        )
    return loaded


__all__ = ["LoadedConfig", "SUPPORTED_METHODS", "load_analysis_configs"]
