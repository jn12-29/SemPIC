from typing import Callable

from transformers import GenerationConfig

from ...model import SupportedModel
from ...prompt import TokenizedPrompt
from ..abc import PrefillResult, PreparedKVMapping, TokenizerType
from ..compact_prefill import CompactPrefillExecutor


def kvpacket_eval(
    model: SupportedModel,
    tokenizer: TokenizerType,
    generation_config: GenerationConfig | None,
    prompt: TokenizedPrompt,
    prepared_kvs: PreparedKVMapping,
    answer: str,
    answer_postprocess_func: Callable[[str, str], tuple[str, str]] | None = None,
    kwargs: dict | None = None,
) -> PrefillResult:
    return CompactPrefillExecutor(model, backend="flex").prefill(
        method_name="kvpacket",
        tokenizer=tokenizer,
        prompt=prompt,
        prepared_kvs=prepared_kvs,
        generation_config=generation_config,
        answer=answer,
        answer_postprocess_func=answer_postprocess_func,
        kwargs=kwargs,
    )

__all__ = [
    "kvpacket_eval",
]
