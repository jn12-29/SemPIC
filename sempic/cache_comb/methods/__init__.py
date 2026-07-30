""" KV Combination Methods """
from ..abc import EvalCombFunc
from .default import no_cache_eval, full_recompute, single_cache
from .no_recompute import no_recompute_eval
from .cache_blend import cache_blend_eval
from .epic import epic_eval
from .sink import sink_eval
from .rand_recompute import rand_recompute_eval
from .kvpacket import kvpacket_eval
from .sempic import sempic_eval
from .sempic_kvpacket import sempic_kvpacket_eval
from .sam_kv import sam_kv_eval
from .a3 import a3_eval

CACHE_COMB_FUNC_DICT: dict[str, EvalCombFunc] = {
    "no_cache": no_cache_eval,
    "full_recompute": full_recompute,
    "no_recompute": no_recompute_eval,
    "cache_blend": cache_blend_eval,
    "epic": epic_eval,
    "sink": sink_eval,
    "rand_recompute": rand_recompute_eval,
    "kvpacket": kvpacket_eval,
    "sempic": sempic_eval,
    "sempic_kvpacket": sempic_kvpacket_eval,
    "sam_kv": sam_kv_eval,
    "a3": a3_eval,
    "single_cache": single_cache,
}

NO_PREP_METHODS = frozenset({"no_cache", "full_recompute", "sink"})
WHOLE_PREFIX_METHODS = frozenset({"single_cache"})
CONTEXT_BLOCK_METHODS = frozenset(CACHE_COMB_FUNC_DICT) - NO_PREP_METHODS - WHOLE_PREFIX_METHODS

assert NO_PREP_METHODS | WHOLE_PREFIX_METHODS | CONTEXT_BLOCK_METHODS == frozenset(CACHE_COMB_FUNC_DICT)

def get_cache_comb_func(name: str) -> EvalCombFunc:
    if name not in CACHE_COMB_FUNC_DICT:
        raise ValueError(f"Unsupported cache combination method: {name}")
    return CACHE_COMB_FUNC_DICT[name]


__all__ = [
    "CACHE_COMB_FUNC_DICT",
    "get_cache_comb_func",
    "NO_PREP_METHODS",
    "WHOLE_PREFIX_METHODS",
    "CONTEXT_BLOCK_METHODS",
]
