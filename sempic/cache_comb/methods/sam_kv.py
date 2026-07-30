import logging
import torch
from typing import Callable
from transformers import GenerationConfig
from torch.nn import functional as F
import itertools
from ..abc import PrefillResult, TokenizerType
# from ...instrument.hf_fix import GenerateFix
from ..recompute_kv import recompute_kv, prepare_pos_embed_and_mask
from ..utils.flops import AutoFlopsCalculator
from ...model import SupportedModel
from ...cache.rotate import rerotate_kv, rerotate_kv_flops
from ...cache import KVCache, concate_kv_caches, get_kv_caches
from ...prompt import TokenSpan, TokenizedPrompt
from ..abc import PreparedKVMapping
from ._prompt_utils import (
    context_parts,
    finish_recomputed_prefill,
    ids_2d,
    require_one_to_one_context_kvs,
    terminal_span,
)

LOGGER = logging.getLogger(__name__)


def _reduce_gqa_query_heads(query: torch.Tensor, n_key_heads: int) -> torch.Tensor:
    n_query_heads = query.size(1)
    if n_query_heads % n_key_heads != 0:
        raise ValueError(
            f"Query heads {n_query_heads} are not compatible with KV heads {n_key_heads}."
        )
    if n_query_heads == n_key_heads:
        return query
    return query.reshape(
        query.size(0),
        n_key_heads,
        n_query_heads // n_key_heads,
        *query.shape[2:],
    ).mean(dim=2)


def _leave_one_out_mean(
    total: torch.Tensor,
    current: torch.Tensor,
    count: int,
) -> torch.Tensor:
    if count == 1:
        return torch.zeros_like(current)
    return (total - current) / (count - 1)


def _sam_peer_parts(
    prompt: TokenizedPrompt,
) -> tuple[list[tuple[int, TokenSpan]], list[int]]:
    peers: list[tuple[int, TokenSpan]] = []
    target_peer_indices: list[int] = []
    for part_index, span in enumerate(prompt.parts[:-1]):
        if span.start == span.end:
            continue
        if span.kind == "context":
            target_peer_indices.append(len(peers))
        peers.append((part_index, span))
    return peers, target_peer_indices


def _ordered_prefix_ids(
    prompt: TokenizedPrompt,
    context_tokens: list[torch.Tensor],
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    context_index = 0
    for span in prompt.parts[:-1]:
        if span.kind == "context":
            parts.append(context_tokens[context_index])
            context_index += 1
        elif span.start != span.end:
            parts.append(ids_2d(prompt, span.start, span.end))
    return torch.cat(parts, dim=1)


def _sam_layer_indices(
    context_physical_indices: list[list[int]],
    context_relative_indices: list[list[int]],
    inline_physical_indices: list[int],
    query_indices: list[int],
) -> tuple[list[int], list[int]]:
    fused_context_indices = [
        context_physical_indices[context_index][relative_index]
        for context_index, relative_indices in enumerate(context_relative_indices)
        for relative_index in relative_indices
    ]
    active_indices = sorted(
        fused_context_indices + inline_physical_indices + query_indices
    )
    return active_indices, fused_context_indices


def _fold_sam_prefix(
    model: SupportedModel,
    prompt: TokenizedPrompt,
    context_caches: list[KVCache],
) -> tuple[KVCache, list[list[int]], list[int], torch.Tensor, int]:
    state: KVCache | None = None
    physical_positions = torch.empty((0,), dtype=torch.long, device=model.device)
    physical_len = 0
    context_physical_indices: list[list[int]] = []
    inline_physical_indices: list[int] = []
    context_index = 0
    flops = 0
    flops_calculator = AutoFlopsCalculator(model)

    for span in prompt.parts[:-1]:
        if span.kind == "context":
            block = context_caches[context_index]
            context_index += 1
            block_len = block[0].key.size(2)
            state = block if state is None else concate_kv_caches([state, block])
            context_physical_indices.append(
                list(range(physical_len, physical_len + block_len))
            )
            physical_positions = torch.cat([
                physical_positions,
                block[0].position_ids.to(model.device).squeeze(0),
            ])
            physical_len += block_len
            continue

        inline_ids = ids_2d(prompt, span.start, span.end).to(model.device)
        if inline_ids.size(1) == 0:
            continue
        inline_positions = torch.arange(
            span.start, span.end, dtype=torch.long, device=model.device
        ).unsqueeze(0)
        inline_physical_indices.extend(
            range(physical_len, physical_len + inline_ids.size(1))
        )
        past = state.to_hf_cache(config=model.config) if state is not None else None
        with torch.no_grad():
            outputs = model.model(
                input_ids=inline_ids,
                past_key_values=past,
                position_ids=inline_positions,
                cache_position=torch.arange(
                    physical_len, physical_len + inline_ids.size(1), device=model.device
                ),
                attention_mask=torch.ones(
                    (1, physical_len + inline_ids.size(1)),
                    dtype=torch.long,
                    device=model.device,
                ),
                use_cache=True,
            )
        if outputs.past_key_values is None:
            raise ValueError("sam_kv Inline prefill returned no KV cache.")
        physical_positions = torch.cat([
            physical_positions,
            inline_positions.squeeze(0),
        ])
        state = KVCache.from_hf_cache(
            outputs.past_key_values,
            position_ids=physical_positions.unsqueeze(0),
        )
        physical_len += inline_ids.size(1)
        flops += flops_calculator.body_flops(
            batch_size=1,
            seq_len=inline_ids.size(1),
            cache_len=physical_len - inline_ids.size(1),
        )

    if context_index != len(context_caches):
        raise ValueError("sam_kv ContextBlock cache count mismatch.")
    if state is None:
        raise ValueError("sam_kv requires at least one ContextBlock.")
    return state, context_physical_indices, inline_physical_indices, physical_positions, flops


def sam_kv_eval(
    model: SupportedModel,
    tokenizer: TokenizerType,
    generation_config: GenerationConfig|None,
    prompt: TokenizedPrompt,
    prepared_kvs: PreparedKVMapping,
    answer: str,
    answer_postprocess_func: Callable[[str, str], tuple[str, str]]|None = None,
    kwargs: dict|None = None
) -> PrefillResult:
    assert model.config.num_hidden_layers is not None
    if kwargs is None:
        raise ValueError("sam_kv_eval requires 'stable_layers', 'num_initial_tokens', and 'num_local_tokens' kwargs")
    else:
        kwargs = kwargs.copy()
        stable_layers: list[int] = kwargs.pop("stable_layers")
        num_initial_tokens: int = kwargs.pop("num_initial_tokens")
        num_local_tokens: int = kwargs.pop("num_local_tokens")
        block_size: int = kwargs.pop("block_size")
        fuse_theta: float = kwargs.pop("fuse_theta")

    if kwargs != {}:
        LOGGER.warning("sam_kv_eval got unexpected kwargs: %s", kwargs)

    require_one_to_one_context_kvs("sam_kv", prompt, prepared_kvs)
    contexts = context_parts(prompt)
    if not contexts:
        raise ValueError("sam_kv requires at least one ContextBlock.")
    doc_tokens = [
        ids_2d(prompt, span.start, span.end).to(model.device)
        for _, span in contexts
    ]
    document_kvs = [
        prepared_kvs[part_index].copy(clone_tensor=True)
        for part_index, _ in contexts
    ]
    terminal = terminal_span(prompt)
    q_ids = ids_2d(prompt, terminal.start, terminal.end).to(model.device)

    peer_parts, target_peer_indices = _sam_peer_parts(prompt)
    context_index_by_part = {
        part_index: context_index
        for context_index, (part_index, _) in enumerate(contexts)
    }
    peer_tokens: list[torch.Tensor] = []
    peer_kvs: list[KVCache] = []
    peer_prefill_flops = 0
    flops_calculator = AutoFlopsCalculator(model)
    for part_index, span in peer_parts:
        tokens = ids_2d(prompt, span.start, span.end).to(model.device)
        peer_tokens.append(tokens)
        if span.kind == "context":
            peer_kvs.append(document_kvs[context_index_by_part[part_index]])
        else:
            peer_kvs.append(get_kv_caches(model, input_ids=tokens)[0])
            peer_prefill_flops += flops_calculator.body_flops(
                batch_size=1,
                seq_len=tokens.size(1),
                cache_len=0,
            )

    sample_len: list[int] = [
        kv[0].position_ids.size(1) for kv in document_kvs
    ]
    # get the query states
    stable_layer_set = set(stable_layers)
    peer_query_states: list[dict[int, torch.Tensor]] = []
    peer_query_flops = 0
    for tokens, peer_kv in zip(peer_tokens, peer_kvs, strict=True):
        peer_len = peer_kv[0].position_ids.size(1)
        hidden_states = model.model.embed_tokens(tokens[:, -num_local_tokens:])
        assert isinstance(hidden_states, torch.Tensor)

        if num_local_tokens >= peer_len:
            token_idx = list(range(peer_len))
        else:
            token_idx = list(range(peer_len - num_local_tokens, peer_len))
    
        pos_embed, recompute_mask = prepare_pos_embed_and_mask(
            model=model,
            hidden_states=hidden_states,
            pos_ids=peer_kv[0].position_ids,
            recompute_indices=token_idx,
        )
        peer_query_state_dict: dict[int, torch.Tensor] = {}

        for layer_index in range(model.config.num_hidden_layers):
            recompute_result = recompute_kv(
                model=model,
                kv_cache=peer_kv,
                hidden_states=hidden_states,
                pos_ids=peer_kv[0].position_ids,
                token_idx=token_idx,
                layer_idx=layer_index,
                update_cache=False,
                pos_embed=pos_embed,
                recompute_mask=recompute_mask,
                return_query_states=True,
            )
            hidden_states = recompute_result["recomputed_hidden_states"]
            query_states = recompute_result["query_states"]
            assert isinstance(query_states, torch.Tensor)
            if layer_index in stable_layer_set:
                peer_query_state_dict[layer_index] = query_states.mean(dim=2)  # [1, num_heads, head_dim]
                peer_query_flops += query_states.numel()
            peer_query_flops += flops_calculator.decoder_layer_flops(
                batch_size=1,
                seq_len=len(token_idx),
                cache_len=peer_len - len(token_idx),
            )
        peer_query_states.append(peer_query_state_dict)
            

    query_len = q_ids.size(1)

    num_head, _ = document_kvs[0][0].key.size(1), document_kvs[0][0].key.size(3)
    num_flops = peer_prefill_flops + peer_query_flops
    
    nope_dim = getattr(model.config, "qk_nope_head_dim", None)
    assert isinstance(nope_dim, int|None)

    for i, (_, span) in enumerate(contexts):
        document_kvs[i] = rerotate_kv(
            document_kvs[i], model.model.rotary_emb, 
            shift=span.start, nope_dim=nope_dim
        )
        num_flops += rerotate_kv_flops(document_kvs[i], nope_dim=nope_dim)

    sparse_kvs, inv_sparse_kvs = get_sparse_kvs(
        kv_caches=document_kvs,
        num_initial_tokens=num_initial_tokens,
        num_local_tokens=num_local_tokens,
    )

    full_sparse_kv, _, _, sparse_positions, fold_flops = _fold_sam_prefix(
        model, prompt, sparse_kvs
    )
    num_flops += fold_flops
    sparse_background_len = full_sparse_kv[0].key.size(2)
    key_head_dim = full_sparse_kv[0].key.size(3)
    value_head_dim = full_sparse_kv[0].value.size(3)

    dummy_query_cache = KVCache.create_dummy(
        num_layers=len(sparse_kvs[0].layers),
        batch_size=1,
        num_heads=num_head,
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
        seq_len=query_len,
        device=model.device,
        dtype=sparse_kvs[0][0].key.dtype,
    )

    full_sparse_kv = concate_kv_caches([full_sparse_kv, dummy_query_cache])

    query_indices_in_sparse = list(
        range(sparse_background_len,  sparse_background_len + query_len)
    )
    terminal_positions = torch.arange(
        terminal.start, terminal.end, device=model.device, dtype=torch.long
    )
    pos_ids_sparse = torch.cat([sparse_positions, terminal_positions])

    hidden_states = model.model.embed_tokens(q_ids)
    assert isinstance(hidden_states, torch.Tensor)

    pos_ids_sparse = pos_ids_sparse.unsqueeze(0)
    pos_embed, recompute_mask = prepare_pos_embed_and_mask(
        model=model,
        hidden_states=hidden_states,
        pos_ids=pos_ids_sparse,
        recompute_indices=query_indices_in_sparse,
    )

    generic_query_states: dict[int, torch.Tensor] = {}

    for layer_index in range(model.config.num_hidden_layers):
        recompute_result = recompute_kv(
            model=model,
            kv_cache=full_sparse_kv,
            hidden_states=hidden_states,
            pos_ids=pos_ids_sparse,
            token_idx=query_indices_in_sparse,
            layer_idx=layer_index,
            update_cache=True,
            pos_embed=pos_embed,
            recompute_mask=recompute_mask,
            return_query_states=True,
        )
        hidden_states = recompute_result["recomputed_hidden_states"]
        query_states = recompute_result["query_states"]

        assert isinstance(query_states, torch.Tensor)
        if layer_index in stable_layer_set:
            generic_query_states[layer_index] = query_states.mean(dim=2)  # [1, num_heads, head_dim]
            num_flops += query_states.numel()

        num_flops += flops_calculator.decoder_layer_flops(
            batch_size=1,
            seq_len=query_len,
            cache_len=sparse_background_len,
        )
    
    query_peer_sim: list[dict[int, torch.Tensor]] = []
    for peer_query_state in peer_query_states:
        weighted_q_peer: dict[int, torch.Tensor] = {}
        for layer_index in stable_layer_set:
            dqs_doc_layer = peer_query_state[layer_index]  # [1, num_heads, head_dim]
            dqs_generic_layer = generic_query_states[layer_index]  # [1, num_heads, head_dim]
            cos_sim = torch.abs(
                torch.nn.functional.cosine_similarity(
                    dqs_doc_layer,
                    dqs_generic_layer,
                    dim=2,
                )
            ) # [1, num_heads]
            weighted_q_peer[layer_index] = dqs_doc_layer * cos_sim.unsqueeze(2)  # [1, num_heads, head_dim]
            vector_count = dqs_doc_layer.numel() // dqs_doc_layer.size(2)
            num_flops += vector_count * (6 * dqs_doc_layer.size(2) + 2)
            num_flops += dqs_doc_layer.numel()
        query_peer_sim.append(weighted_q_peer)

    query_doc: list[dict[int, torch.Tensor]] = [{} for _ in document_kvs] # [num_documents, num_stable_layers]
    for j in stable_layer_set:
        layer_queries = [peer[j] for peer in query_peer_sim]
        layer_sum = torch.stack(layer_queries).sum(dim=0)
        num_flops += (len(layer_queries) - 1) * layer_sum.numel()
        dqs_generic_layer = generic_query_states[j]  # [1, num_heads, head_dim]
        for target_index, peer_index in enumerate(target_peer_indices):
            peer_mean = _leave_one_out_mean(
                layer_sum,
                layer_queries[peer_index],
                len(layer_queries),
            )
            query_doc[target_index][j] = peer_mean + dqs_generic_layer
            if len(layer_queries) > 1:
                num_flops += 2 * peer_mean.numel()
            num_flops += peer_mean.numel()

    p_doc_layer: list[dict[int, float]] = []
    doc_inner_list: list[dict[int, torch.Tensor|None]] = []
    for i in range(len(document_kvs)):
        p_dict: dict[int, float] = {}
        doc_inner_dict: dict[int, torch.Tensor|None] = {}
        for j in stable_layer_set:
            inv_kv_i = inv_sparse_kvs[i]
            if inv_kv_i is None or inv_kv_i[j].key.numel() == 0:
                p_dict[j] = 0.0
                doc_inner_dict[j] = None
            else:
                k_i_j_anc = sparse_kvs[i][j].key.mean(dim=2) # [1, num_heads, head_dim]
                k_i_j_doc = inv_kv_i[j].key # [1, num_heads, seq_len, head_dim]
                num_flops += sparse_kvs[i][j].key.numel()
                num_flops += k_i_j_doc.numel()
                k_i_j_doc = interleaved_mean(k_i_j_doc, dim=2, block_size=block_size) # [1, num_heads, num_block, head_dim]
                qd_ij = query_doc[i][j] # [1, num_heads, head_dim]
                if qd_ij.size(1) != k_i_j_anc.size(1):
                    num_flops += qd_ij.numel()
                    qd_ij = _reduce_gqa_query_heads(qd_ij, k_i_j_anc.size(1))
                anc_inner = torch.sum(k_i_j_anc * qd_ij) # []
                doc_inner = torch.sum(k_i_j_doc * qd_ij.unsqueeze(2), dim=(0, 1, 3)) # [num_block]
                num_flops += 2 * k_i_j_anc.numel() - 1
                num_flops += 2 * k_i_j_doc.numel() - k_i_j_doc.size(2)
                max_inner = torch.max(doc_inner)
                min_inner = torch.min(doc_inner)

                if min_inner.item() < anc_inner.item() <= max_inner.item():
                    p_doc_i_j = (max_inner.item() - anc_inner.item()) / (max_inner.item() - min_inner.item() + 1e-8)
                else:
                    p_doc_i_j = 0.0
                p_dict[j] = p_doc_i_j
                doc_inner_dict[j] = doc_inner
        doc_inner_list.append(doc_inner_dict)
        p_doc_layer.append(p_dict)

    recompute_indices: list[dict[int, list[int]]] = []
    for i in range(len(document_kvs)):
        p_list = list(p_doc_layer[i].values())
        recompute_ratio = sum(p_list) / len(p_list)
        indices_dict: dict[int, list[int]] = {}
        for j in stable_layer_set:
            doc_inner = doc_inner_list[i][j]
            cache_len = sample_len[i]
            if doc_inner is None:
                indices_dict[j] = list(range(cache_len))
            else:
                num_recompute_blocks = int(doc_inner.shape[0] * recompute_ratio)
                top_k_indices = torch.topk(doc_inner, k=num_recompute_blocks).indices.tolist()
                indices_dict[j] = get_recompute_indices(
                    top_k_indices,
                    block_size,
                    num_initial_tokens,
                    num_local_tokens,
                    cache_len
                )
        recompute_indices.append(indices_dict)

    abs_indices_doc_max: list[list[int]] = [] # max indices to recompute absolute in doc
    or_indices_new: list[dict[int, list[int]]] = [] # or indices relative to new cache in doc

    for item in recompute_indices:
        _, or_indices_dict_new, max_indices = get_or_indices(item)
        abs_indices_doc_max.append(max_indices)
        or_indices_new.append(or_indices_dict_new)
    
    selected_contexts: list[KVCache] = []
    selected_tokens: list[torch.Tensor] = []
    for cache, tokens, selected in zip(
        document_kvs, doc_tokens, abs_indices_doc_max, strict=True
    ):
        selected_tensor = torch.tensor(
            selected, dtype=torch.long, device=model.device
        )
        selected_contexts.append(cache.select_seq(selected_tensor))
        selected_tokens.append(tokens[:, selected_tensor])

    sparse_kv, context_physical_indices, inline_physical_indices, prefix_positions, fold_flops = _fold_sam_prefix(
        model, prompt, selected_contexts
    )
    num_flops += fold_flops
    physical_prefix_len = sparse_kv[0].key.size(2)

    dummy_query_cache = KVCache.create_dummy(
        num_layers=len(sparse_kvs[0].layers),
        batch_size=1,
        num_heads=num_head,
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
        seq_len=query_len,
        device=model.device,
        dtype=sparse_kvs[0][0].key.dtype,
    )

    sparse_kv = concate_kv_caches([sparse_kv, dummy_query_cache])
    hidden_states = model.model.embed_tokens(
        torch.cat([
            _ordered_prefix_ids(prompt, selected_tokens).to(model.device),
            q_ids,
        ], dim=1)
    )
    assert isinstance(hidden_states, torch.Tensor)
    pos_ids = torch.cat([prefix_positions, terminal_positions]).unsqueeze(0)
    recompute_token_indices = list(range(physical_prefix_len))
    query_indices = list(range(
        physical_prefix_len, physical_prefix_len + query_len
    ))
    recompute_token_indices += query_indices

    for layer in range(model.config.num_hidden_layers):
        or_indices_all_doc = [item.get(layer, []) for item in or_indices_new]
        new_recompute_token_indices, context_indices_for_layer = _sam_layer_indices(
            context_physical_indices,
            or_indices_all_doc,
            inline_physical_indices,
            query_indices,
        )

        if len(new_recompute_token_indices) == 0:
            break

        recompute_token_indices_map = {val: i for i, val in enumerate(recompute_token_indices)}
        new_relative_indices = [recompute_token_indices_map[item] for item in new_recompute_token_indices]
        hidden_states = hidden_states[:, new_relative_indices, :]
        recompute_token_indices = new_recompute_token_indices

        hidden_states = recompute_kv(
            model=model,
            kv_cache=sparse_kv,
            hidden_states=hidden_states,
            pos_ids=pos_ids,
            token_idx=recompute_token_indices,
            layer_idx=layer,
            update_cache=True,
            fuse_indices=context_indices_for_layer,
            fuse_theta=fuse_theta
        )["recomputed_hidden_states"]
        num_flops += flops_calculator.decoder_layer_flops(
            batch_size=1, seq_len=len(recompute_token_indices),
            cache_len=physical_prefix_len+query_len-len(recompute_token_indices)
        )
        fused_elements = len(context_indices_for_layer) * (
            sparse_kv[layer].key.size(0)
            * sparse_kv[layer].key.size(1)
            * sparse_kv[layer].key.size(3)
            + sparse_kv[layer].value.size(0)
            * sparse_kv[layer].value.size(1)
            * sparse_kv[layer].value.size(3)
        )
        num_flops += 3 * fused_elements

    return finish_recomputed_prefill(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        cache=sparse_kv,
        terminal_hidden_states=hidden_states[:, -query_len:, :],
        terminal_ids=q_ids,
        prefix_physical_len=physical_prefix_len,
        flops=num_flops,
    )



def get_sparse_kvs(
    kv_caches: list[KVCache],
    num_initial_tokens: int,
    num_local_tokens: int,
) -> tuple[list[KVCache], list[KVCache|None]]:
    sparse_kvs: list[KVCache] = []
    inv_sparse_kvs: list[KVCache|None] = []
    for kv_cache in kv_caches:
        cache_len = kv_cache[0].key.size(2)
        if cache_len <= num_initial_tokens + num_local_tokens:
            sparse_kvs.append(kv_cache.select_seq(
                torch.arange(0, cache_len, device=kv_cache[0].key.device, dtype=torch.long)
            ))
            inv_sparse_kvs.append(None)
        else:
            device = kv_cache[0].key.device
            initial_indices = torch.arange(
                0, num_initial_tokens, device=device, dtype=torch.long
            )
            local_indices = torch.arange(
                cache_len - num_local_tokens,
                cache_len,
                device=device,
                dtype=torch.long,
            )
            selected_indices = torch.cat([initial_indices, local_indices], dim=0)
            sparse_kvs.append(
                kv_cache.select_seq(selected_indices)
            )
            inv_sparse_kvs.append(
                kv_cache.select_seq(
                    torch.arange(
                        num_initial_tokens,
                        cache_len - num_local_tokens,
                        device=device,
                        dtype=torch.long,
                    )
                )
            )
    return sparse_kvs, inv_sparse_kvs


def interleaved_mean(tensor: torch.Tensor, dim: int, block_size: int):
    """
    Computes the mean of a tensor along a specific dimension in interleaved blocks.
    
    Args:
        tensor (torch.Tensor): The input tensor.
        dim (int): The dimension to perform the mean on.
        block_size (int): The size of the stride/window to average.
        
    Returns:
        torch.Tensor: The tensor with the reduced dimension.
    """
    ndims = tensor.dim()
    dim = dim % ndims

    dims_order = list(range(ndims))
    dims_order.append(dims_order.pop(dim))
    
    x = tensor.permute(*dims_order)
    original_permuted_shape = x.shape
    x = x.reshape(-1, 1, original_permuted_shape[-1])
    x = F.avg_pool1d(x, kernel_size=block_size, stride=block_size, ceil_mode=True)
    new_shape = list(original_permuted_shape)
    new_shape[-1] = x.shape[-1]
    x = x.view(*new_shape)
    inverse_dims_order = [0] * ndims
    for i, p in enumerate(dims_order):
        inverse_dims_order[p] = i
        
    return x.permute(*inverse_dims_order)


def get_recompute_indices(
    block_indices: list[int],
    block_size: int,
    num_initial_tokens: int,
    num_local_tokens: int,
    doc_len: int
) -> list[int]:
    block_indices.sort()
    indices: list[int] = list(range(num_initial_tokens))

    for block_index in block_indices:
        start_index = num_initial_tokens + block_index * block_size
        end_index = min(num_initial_tokens + (block_index + 1 ) * block_size, doc_len - num_local_tokens)
        indices.extend(range(start_index, end_index))

    indices.extend(range(doc_len - num_local_tokens, doc_len))
    return indices


def get_or_indices(indices_dict: dict[int, list[int]]):
    max_indices = list(set(itertools.chain.from_iterable(list(indices_dict.values()))))
    max_indices.sort()

    index_map = {val: i for i, val in enumerate(max_indices)}
    layer_indices = list(indices_dict.keys())
    layer_indices.sort(reverse=True)

    or_indices_dict_new: dict[int, list[int]] = {} # local index
    indices_dict_new: dict[int, list[int]] = {} # local index
    indices_set: set[int] = set()
    for layer_index in layer_indices:
        indices_set = indices_set | set(indices_dict[layer_index])
        indices_list = list(indices_set)
        indices_list.sort()
        or_indices_dict_new[layer_index] = [index_map[item] for item in indices_list]
        indices_dict_new[layer_index] = [index_map[item] for item in indices_dict[layer_index]]

    for layer_index in range(layer_indices[0], -1, -1):
        if layer_index not in or_indices_dict_new:
            or_indices_dict_new[layer_index] = or_indices_dict_new[layer_index+1]

    return indices_dict_new, or_indices_dict_new, max_indices
