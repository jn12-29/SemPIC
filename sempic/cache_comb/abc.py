from collections.abc import Mapping
from typing import Callable, Protocol, NamedTuple, TypeAlias
import torch
from transformers import GenerationConfig
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.cache_utils import Cache
from ..cache import KVCache
from ..model import SupportedModel
from ..prompt import TokenizedPrompt

TokenizerType = PreTrainedTokenizer | PreTrainedTokenizerFast
PromptPartIndex: TypeAlias = int
PreparedKVMapping: TypeAlias = Mapping[PromptPartIndex, KVCache]

class PrefillResult(NamedTuple):
    logits: torch.Tensor
    past_key_values: Cache
    generation_input_ids: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: torch.Tensor
    flops: int


class EvalCombFunc(Protocol):
    def __call__(
        self,
        model: SupportedModel,
        tokenizer: TokenizerType,
        generation_config: GenerationConfig|None,
        prompt: TokenizedPrompt,
        prepared_kvs: PreparedKVMapping,
        answer: str,
        answer_postprocess_func: Callable[[str, str], tuple[str, str]]|None = None,
        kwargs: dict|None = None
    ) -> PrefillResult:
        """
        Combine runtime KV artifacts for one canonical token stream.

        ``prepared_kvs`` is external runtime state keyed by prompt part index.
        Prompt part kinds remain semantic input and are never inferred from the
        order of mutable cache lists.
        """
        ...
