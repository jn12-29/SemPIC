""" Custom KV Cache class """
import torch
from torch.utils.checkpoint import checkpoint
from transformers import LlamaForCausalLM, PretrainedConfig, Qwen3ForCausalLM
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import CausalLMOutputWithPast
from kvpress import BasePress
from contextlib import nullcontext
from typing import Callable, ContextManager, Iterator
from warnings import warn
from ..model import SupportedModel
from .abc import KeyValue, KVDim, PosIDDim, KeyValueDict
from .kv_cache_state_dict import KVCacheStateDict
from .quantization import QuantizedTensor, dequantize_tensor
from .compress import KeepIndex

__all__ = [
    "KVCache",
    "concate_kv_caches",
    "get_kv_caches",
    "get_kv_caches_with_grad",
]

class KVCache:
    """
    Cache for storing key-value pairs for multiple layers.

    For each layer, it stores a list of KeyValue pairs along the sequence dimension.
    """
    def __init__(self):
        self._cache: dict[int, list[KeyValue]] = {}


    @property
    def layers(self) -> list[int]:
        """ Get the list of layer indices in the cache """
        return sorted(self._cache.keys())


    def get_seq_length(self, layer: int) -> int:
        """Return a layer's sequence length without consolidating its chunks."""
        kv_list = self._cache.get(layer)
        if not kv_list:
            raise KeyError(f"Layer {layer} not found in KVCache.")
        return sum(key_value.key.size(KVDim["seq"]) for key_value in kv_list)


    def update(self, layer: int, key_value: KeyValue):
        """ Add a new KeyValue pair to the cache for a specific layer
        The KeyValue pair will be added to the list of KeyValue pairs for that layer
        """
        if layer not in self._cache:
            self._cache[layer] = []
        self._cache[layer].append(key_value)


    def _consolidate(self, layer: int) -> None|KeyValue:
        """
        Consolidate the list of KeyValue pairs for a layer into a single KeyValue pair

        If there are multiple KeyValue pairs, concatenate them along the seq dimension.
        This will also change the KV in the cache to be the consolidated version

        Args:
            - layer: Layer index to consolidate
        Returns:
            - KeyValue: Consolidated KeyValue pair, or ``None`` if layer not found
        """
        kv_list = self._cache.get(layer, [])

        if len(kv_list) == 0:
            return None

        if len(kv_list) == 1:
            # If there's 0 or 1 KeyValue pair, no need to consolidate
            return kv_list[0]

        keys = torch.cat([kv["key"] for kv in kv_list], dim=KVDim["seq"])
        values = torch.cat([kv["value"] for kv in kv_list], dim=KVDim["seq"])
        position_ids = torch.cat([kv["position_ids"] for kv in kv_list], dim=PosIDDim["seq"])
        key_value = KeyValue(key=keys, value=values, position_ids=position_ids)
        self._cache[layer] = [key_value]

        return key_value


    def get_consolidated(self) -> "KVCache":
        """ Get a dictionary of consolidated KeyValue pairs for all layers """
        # consolidated_kv = {}
        for layer in self.layers:
            self._consolidate(layer)
        return self


    def __getitem__(self, index: int) -> KeyValue:
        """ Get the consolidated KeyValue pair for a specific layer """
        key_value = self._consolidate(index)
        if key_value is None:
            raise KeyError(f"Layer {index} not found in KVCache.")
        return key_value


    def __setitem__(self, index: int, value: KeyValue|list[KeyValue]) -> None:
        """ Set the KeyValue pair or list of KeyValue pairs for a specific layer """
        if index not in self._cache:
            self._cache[index] = []
        if isinstance(value, KeyValue):
            self._cache[index] = [value]
        elif isinstance(value, list):
            self._cache[index] = value
        else:
            raise ValueError("Value must be a KeyValue or a list of KeyValue.")


    def copy(self, clone_tensor: bool=False) -> 'KVCache':
        """ Create a copy of the KVCache. If clone_tensor is True, clone the tensors. """
        new_cache = KVCache()
        for layer, kv_list in self._cache.items():
            if clone_tensor:
                new_kv_list = [
                    KeyValue(
                        key=kv["key"].clone(),
                        value=kv["value"].clone(),
                        position_ids=kv["position_ids"].clone()
                    ) for kv in kv_list
                ]
            else:
                new_kv_list = [
                    KeyValue(
                        key=kv["key"],
                        value=kv["value"],
                        position_ids=kv["position_ids"]
                    ) for kv in kv_list
                ]
            new_cache._cache[layer] = new_kv_list
        return new_cache


    def __contains__(self, index: int) -> bool:
        """ Check if a layer index exists in the cache """
        return index in self._cache


    def __iter__(self) -> Iterator[KeyValue]:
        """ Iterate over the layer indices in the cache """
        for layer in self.layers:
            yield self[layer]


    @staticmethod
    def from_hf_cache(hf_cache: Cache, position_ids: torch.Tensor|None=None) -> 'KVCache':
        """ Convert a HuggingFace Cache to a KVCache """
        kv_cache = KVCache()
        for layer_index, layer in enumerate(hf_cache.layers):
            key = layer.keys
            value = layer.values
            assert isinstance(key, torch.Tensor)
            assert isinstance(value, torch.Tensor)

            if position_ids is None:
                position_ids_t = torch.arange(
                    key.size(KVDim["seq"]), dtype=torch.long, device=key.device
                ).unsqueeze(0).to(key.device)
            else:
                position_ids_t = position_ids.to(key.device)
            key_value = KeyValue(key=key, value=value, position_ids=position_ids_t)
            kv_cache.update(layer_index, key_value)
        return kv_cache


    def to_hf_cache(self, config: PretrainedConfig|None=None) -> DynamicCache:
        """
        Convert the KVCache to a HuggingFace Cache
        Args:
            - config: Optional PretrainedConfig for the DynamicCache
        Returns:
            - DynamicCache: The converted HuggingFace Cache

        Warning:
            - The huggingface Cache does not store position_ids. The position_ids
                must be managed separately if needed.
            - The start position for the 'generate' method in transformers is based on the 
                length of the kv_cache and the input_ids. The position will be incorrect
                if the kv_cache is compressed. In that case, the position_ids should be
                used to determine the correct position.
        """
        hf_cache = DynamicCache(config=config)
        sorted_layers = sorted(self.layers)

        for layer_index in sorted_layers:
            kv = self._consolidate(layer_index)
            assert kv is not None
            hf_cache.update(
                key_states=kv.key,
                value_states=kv.value,
                layer_idx=layer_index
            )
        return hf_cache


    def state_dict(self, compact: bool=True) -> KVCacheStateDict:
        """ Convert the KVCache to a state dictionary """
        layers: dict[int, KeyValueDict|None] = {}

        position_ids: torch.Tensor|None = None
        key_list: list[torch.Tensor] = []
        value_list: list[torch.Tensor] = []

        self.get_consolidated()

        if self.layers == []:
            return KVCacheStateDict(
                layers={},
                compact_kv=None,
                position_ids=None,
            )
    
        first_layer_kv = self[self.layers[0]]
        key_shape = first_layer_kv.key.shape
        position_ids = first_layer_kv.position_ids

        if compact:
            for layer in self.layers:
                kv = self._consolidate(layer)
                assert kv is not None
                if not torch.equal(position_ids, kv.position_ids):
                    warn("Position IDs differ across layers; cannot create compact_kv.")
                    compact = False
                    break
                
                if kv.key.shape != key_shape or kv.value.shape != key_shape:
                    warn("Key/Value shapes differ across layers; cannot create compact_kv.")
                    compact = False
                    break
    
        if compact:
            for layer in self.layers:
                kv = self._consolidate(layer)
                assert kv is not None
                layers[layer] = None  # Indicate compact_kv is used
                key_list.append(kv.key)
                value_list.append(kv.value)
            compact_key = torch.stack(key_list, dim=0)  # [num_layers, batch, num_heads, seq_len, head_dim]
            compact_value = torch.stack(value_list, dim=0)  # [num_layers, batch, num_heads, seq_len, head_dim]
            compact_kv = torch.stack([compact_key, compact_value], dim=0)  # [2, num_layers, batch, num_heads, seq_len, head_dim]

            return KVCacheStateDict(
                layers=layers,
                compact_kv=compact_kv,
                position_ids=position_ids,
            )
        else:
            for layer in self.layers:
                kv = self._consolidate(layer)
                assert kv is not None
                key_value_dict: KeyValueDict = {
                    "key": kv.key,
                    "value": kv.value,
                    "position_ids": kv.position_ids
                }
                layers[layer] = key_value_dict

            return KVCacheStateDict(
                layers=layers,
                compact_kv=None,
                position_ids=None,
            )


    @staticmethod
    def from_state_dict(state_dict: KVCacheStateDict) -> 'KVCache':
        """ Create a KVCache from a state dictionary """
        kv_cache = KVCache()

        layers = state_dict["layers"]

        num_layers = len(layers)
        compact_kv = state_dict["compact_kv"]
        position_ids = state_dict["position_ids"]

        if compact_kv is not None:
            if position_ids is None:
                raise ValueError("position_ids must be provided when using compact_kv.")

            if isinstance(compact_kv, QuantizedTensor):
                if compact_kv.device.type == "cpu":
                    warn("Dequantizing compact_kv on CPU; this may be slow.")
                compact_kv = dequantize_tensor(compact_kv)

            compact_key = compact_kv[0]  # [num_layers, batch, num_heads, seq_len, head_dim]
            compact_value = compact_kv[1]  # [num_layers, batch, num_heads, seq_len, head_dim]

            if compact_key.size(0) != num_layers or compact_value.size(0) != num_layers:
                raise ValueError("Mismatch between number of layers and compact_kv size.")

            for layer in range(num_layers):
                key = compact_key[layer]
                value = compact_value[layer]
                key_value = KeyValue(
                    key=key,
                    value=value,
                    position_ids=position_ids
                )
                kv_cache.update(layer, key_value)
            return kv_cache
        else:
            for layer, kv_dict in state_dict["layers"].items():
                if kv_dict is None:
                    raise ValueError(f"KeyValue dict for layer {layer} is None in non-compact state_dict.")
                key_value = KeyValue(
                    key=kv_dict["key"],
                    value=kv_dict["value"],
                    position_ids=kv_dict["position_ids"]
                )
                kv_cache.update(layer, key_value)
            return kv_cache


    @staticmethod
    def create_dummy(
        num_layers: int,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        key_head_dim: int,
        value_head_dim: int,
        position_ids: torch.Tensor|None=None,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> 'KVCache':
        """ Create a dummy KVCache with random tensors for testing """
        kv_cache = KVCache()
        for layer in range(num_layers):
            key = torch.zeros(
                (batch_size, num_heads, seq_len, key_head_dim),
                dtype=dtype,
                device=device
            )
            value = torch.zeros(
                (batch_size, num_heads, seq_len, value_head_dim),
                dtype=dtype,
                device=device
            )
            if position_ids is None:
                position_ids_t = torch.arange(
                    seq_len, dtype=torch.long, device=device
                ).unsqueeze(0).to(device)
            else:
                position_ids_t = position_ids.to(device)
            key_value = KeyValue(
                key=key,
                value=value,
                position_ids=position_ids_t
            )
            kv_cache.update(layer, key_value)
        return kv_cache


    def to(self, device: torch.device, non_blocking: bool=False) -> 'KVCache':
        """ Move all tensors in the KVCache to the specified device """
        new_cache = KVCache()
        for layer, kv_list in self._cache.items():
            new_kv_list = [
                KeyValue(
                    key=kv["key"].to(device, non_blocking=non_blocking),
                    value=kv["value"].to(device, non_blocking=non_blocking),
                    position_ids=kv["position_ids"].to(device, non_blocking=non_blocking)
                ) for kv in kv_list
            ]
            new_cache._cache[layer] = new_kv_list
        return new_cache


    def select_seq(self, indices: torch.Tensor) -> 'KVCache':
        """ Select specific sequence positions from the KVCache based on indices """
        new_cache = KVCache()
        for layer in self.layers:
            kv = self[layer]
            if kv.position_ids.size(PosIDDim["seq"]) != kv.key.size(KVDim["seq"]):
                raise ValueError(
                    "Cannot select KV positions without a one-to-one position mapping."
                )
            selected_key = torch.index_select(kv.key, KVDim["seq"], indices)
            selected_value = torch.index_select(kv.value, KVDim["seq"], indices)
            selected_position_ids = torch.index_select(
                kv.position_ids,
                PosIDDim["seq"],
                indices,
            )
            key_value = KeyValue(
                key=selected_key,
                value=selected_value,
                position_ids=selected_position_ids,
            )
            new_cache.update(layer, key_value)
        return new_cache


def concate_kv_caches(caches: list[KVCache]) -> KVCache:
    """
    Concatenate multiple KVCache instances along the seq dimension

    Args:
        - caches: List of KVCache instances to concatenate
    Returns:
        - KVCache: A new KVCache instance with concatenated KeyValue pairs
    """
    if len(caches) == 0:
        raise ValueError("No KVCache instances to concatenate.")

    new_cache = KVCache()
    
    for cache in caches:
        for layer, kv_list in cache._cache.items():
            if layer not in new_cache._cache:
                new_cache._cache[layer] = []
            new_cache._cache[layer].extend(kv_list)

    return new_cache


def _get_model_device(model: SupportedModel) -> torch.device:
    device = getattr(model, "device", None)
    if isinstance(device, torch.device):
        return device
    return next(model.parameters()).device


def _get_causal_lm(model: SupportedModel) -> SupportedModel:
    causal_lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    if not isinstance(causal_lm, (LlamaForCausalLM, Qwen3ForCausalLM)):
        raise TypeError(f"Unsupported model type: {type(causal_lm).__name__}.")
    return causal_lm


def _get_layerwise_checkpointed_cache_tensors(
    model: SupportedModel,
    input_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor | None,
    forward_context_factory: Callable[[], ContextManager[None]] | None,
) -> tuple[torch.Tensor, ...]:
    causal_lm = _get_causal_lm(model)
    body = causal_lm.model
    cache_position = torch.arange(input_embeds.size(1), device=input_embeds.device)
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    mask_cache = DynamicCache(config=causal_lm.config)
    attention_masks = {
        "full_attention": create_causal_mask(
            config=causal_lm.config,
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=mask_cache,
            position_ids=position_ids,
        ),
    }
    if isinstance(causal_lm, Qwen3ForCausalLM) and body.has_sliding_layers:
        attention_masks["sliding_attention"] = create_sliding_window_causal_mask(
            config=causal_lm.config,
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=mask_cache,
            position_ids=position_ids,
        )

    position_embeddings = body.rotary_emb(input_embeds, position_ids)
    hidden_states = input_embeds
    cache_tensors: list[torch.Tensor] = []
    for layer_index, decoder_layer in enumerate(body.layers[: causal_lm.config.num_hidden_layers]):
        attention_type = getattr(decoder_layer, "attention_type", "full_attention")
        layer_attention_mask = attention_masks[attention_type]

        def make_layer_forward(
            current_layer=decoder_layer,
            current_layer_index=layer_index,
            current_attention_mask=layer_attention_mask,
        ):
            def layer_forward(layer_hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                layer_cache = DynamicCache(config=causal_lm.config)
                forward_context = (
                    forward_context_factory()
                    if forward_context_factory is not None
                    else nullcontext()
                )
                with forward_context:
                    next_hidden_states = current_layer(
                        layer_hidden_states,
                        attention_mask=current_attention_mask,
                        position_ids=position_ids,
                        past_key_values=layer_cache,
                        use_cache=True,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )
                layer = layer_cache.layers[current_layer_index]
                assert isinstance(layer.keys, torch.Tensor) and isinstance(layer.values, torch.Tensor)
                return next_hidden_states, layer.keys, layer.values

            return layer_forward

        hidden_states, layer_key, layer_value = checkpoint(
            make_layer_forward(),
            hidden_states,
            use_reentrant=False,
        )
        cache_tensors.extend((layer_key, layer_value))

    return tuple(cache_tensors)


def _get_kv_caches_impl(
    model: SupportedModel,
    input_ids: torch.Tensor|None = None,
    input_embeds: torch.Tensor| None = None,
    attention_mask: torch.Tensor|None = None,
    position_ids: torch.Tensor|None = None,
    compressor: BasePress|None=None,
    indices_to_keep: list[int]|None=None,
    enable_grad: bool=False,
    checkpoint_grad: bool=False,
    forward_context_factory: Callable[[], ContextManager[None]] | None = None,
) -> list[KVCache]:
    """
    Get the KVCache using the provided model and inputs, with optional compression and quantization 
    
    Args:
        - model: LlamaForCausalLM model to generate the KV cache
        - input_ids: Input token IDs
        - input_embeds: Input embeddings
        - position_ids: Positional IDs for the inputs
        - compressor: Optional BasePress compressor to compress the KV cache
        # - quantization: Optional quantization level (2, 4, or 8 bits)
    Returns:
        - KVCache: The generated (and possibly compressed/quantized) KV cache
    """
    if input_ids is None:
        if input_embeds is None:
            raise ValueError("Either input_ids or input_embeds must be provided.")
        input_len = input_embeds.size(1)
        batch_size = input_embeds.size(0)
    else:
        if checkpoint_grad:
            raise ValueError("checkpoint_grad requires input_embeds.")
        input_len = input_ids.size(1)
        batch_size = input_ids.size(0)

    model_device = _get_model_device(model)

    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, input_len), dtype=torch.int, device=model_device
        )

    if batch_size > 1 and compressor is not None:
        raise ValueError("Compressor currently does not support batch_size > 1.")

    if compressor is not None and not attention_mask.all().item():
        raise ValueError("Compressor currently does not support attention masks with False values.")

    pos_ids_list: list[torch.Tensor] = []

    if position_ids is None:
        pos_ids = torch.arange(
            input_len, dtype=torch.long, device=model_device
        )
        for i in range(batch_size):
            pos_ids_list.append(pos_ids[attention_mask[i].bool()].unsqueeze(0))
    else:
        for i in range(batch_size):
            pos_ids = position_ids[i].to(model_device)
            pos_ids_list.append(pos_ids[attention_mask[i].bool()].unsqueeze(0))

    grad_context = torch.enable_grad() if enable_grad else torch.no_grad()

    def forward_cache(
        *,
        input_ids_arg: torch.Tensor|None,
        input_embeds_arg: torch.Tensor|None,
    ) -> tuple[torch.Tensor, ...]:
        forward_context = (
            forward_context_factory()
            if forward_context_factory is not None
            else nullcontext()
        )
        with forward_context:
            result: CausalLMOutputWithPast = model(
                input_ids=input_ids_arg,
                inputs_embeds=input_embeds_arg,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                logits_to_keep=1,
            )
        cache = result.past_key_values
        assert isinstance(cache, Cache)
        cache_tensors: list[torch.Tensor] = []
        for layer in cache.layers:
            assert isinstance(layer.keys, torch.Tensor) and isinstance(layer.values, torch.Tensor)
            cache_tensors.extend([layer.keys, layer.values])
        return tuple(cache_tensors)

    if checkpoint_grad and not enable_grad:
        raise ValueError("checkpoint_grad requires enable_grad=True.")
    if checkpoint_grad and compressor is not None:
        raise ValueError("checkpoint_grad is not supported with compressors.")
    if checkpoint_grad and indices_to_keep is not None:
        raise ValueError("checkpoint_grad is not supported with indices_to_keep.")

    with grad_context, compressor(model=model) if compressor is not None else nullcontext(), KeepIndex(indices_to_keep):
        if checkpoint_grad:
            assert input_embeds is not None
            cache_tensors = _get_layerwise_checkpointed_cache_tensors(
                model,
                input_embeds,
                attention_mask,
                position_ids,
                forward_context_factory,
            )
        else:
            cache_tensors = forward_cache(input_ids_arg=input_ids, input_embeds_arg=input_embeds)

    caches: list[KVCache] = []
    for b in range(batch_size):
        attn_mask = attention_mask[b].bool()
        attn_mask_all_true = attn_mask.all().item()
        pos_ids = pos_ids_list[b]

        kv_cache = KVCache()
        for i in range(0, len(cache_tensors), 2):
            layer_index = i // 2
            layer_keys = cache_tensors[i]
            layer_values = cache_tensors[i + 1]
            if attn_mask_all_true:
                key = layer_keys[b : b + 1, :, :, :]
                value = layer_values[b : b + 1, :, :, :]
            else:
                key = layer_keys[b : b + 1, :, attn_mask, :]
                value = layer_values[b : b + 1, :, attn_mask, :]
            
            key_value = KeyValue(
                key=key,
                value=value,
                position_ids=pos_ids
            )
            kv_cache.update(layer_index, key_value)
        caches.append(kv_cache)

    return caches


def get_kv_caches(
    model: SupportedModel,
    input_ids: torch.Tensor|None = None,
    input_embeds: torch.Tensor| None = None,
    attention_mask: torch.Tensor|None = None,
    position_ids: torch.Tensor|None = None,
    compressor: BasePress|None=None,
    indices_to_keep: list[int]|None=None,
) -> list[KVCache]:
    """
    Get the KVCache using the provided model and inputs without tracking gradients.
    """
    return _get_kv_caches_impl(
        model=model,
        input_ids=input_ids,
        input_embeds=input_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        compressor=compressor,
        indices_to_keep=indices_to_keep,
        enable_grad=False,
    )


def get_kv_caches_with_grad(
    model: SupportedModel,
    input_ids: torch.Tensor|None = None,
    input_embeds: torch.Tensor| None = None,
    attention_mask: torch.Tensor|None = None,
    position_ids: torch.Tensor|None = None,
    compressor: BasePress|None=None,
    indices_to_keep: list[int]|None=None,
    checkpoint_grad: bool=False,
    forward_context_factory: Callable[[], ContextManager[None]] | None = None,
) -> list[KVCache]:
    """
    Get the KVCache while preserving the autograd graph from the model forward pass.

    This is intended for trainable document KV paths where the returned tensors must
    retain gradients back to the component that produced them.
    """
    return _get_kv_caches_impl(
        model=model,
        input_ids=input_ids,
        input_embeds=input_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        compressor=compressor,
        indices_to_keep=indices_to_keep,
        enable_grad=True,
        checkpoint_grad=checkpoint_grad,
        forward_context_factory=forward_context_factory,
    )
