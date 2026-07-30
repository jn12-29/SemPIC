from functools import partial

from ..model import SupportedModel
from .abc import EvalCombFunc
from .compact_prefill import CompactPrefillExecutor
from .methods import get_cache_comb_func


_COMPACT_PREFILL_METHODS = frozenset({
    "no_recompute",
    "kvpacket",
    "sempic",
    "sempic_kvpacket",
})


def build_cache_comb_executor(name: str, model: SupportedModel) -> EvalCombFunc:
    """Bind one cache-combination executor before request timing begins."""
    fallback = get_cache_comb_func(name)
    if (
        name not in _COMPACT_PREFILL_METHODS
        or not fallback.__module__.startswith("sempic.cache_comb.methods.")
    ):
        return fallback

    executor = CompactPrefillExecutor(model, backend="flex")
    bound = partial(executor.prefill, method_name=name)
    bound.warmup = partial(executor.warm_request, method_name=name)  # type: ignore[attr-defined]
    return bound


__all__ = ["build_cache_comb_executor"]
