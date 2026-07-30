# Prompt Sequence Contract

## Purpose

Training and evaluation represent a prompt as an ordered sequence of two essential kinds of text:

- `Inline`: ordinary text that participates in normal causal prefill.
- `ContextBlock`: a separately addressable context unit whose preparation is chosen by the active method.

This representation is semantic input. It does not encode cache producers, compression state, devices, runtime roles, or an execution graph.

## Semantic Types

```python
PromptPart[T] = Inline[T] | ContextBlock[T]
PromptSequence[T] = tuple[PromptPart[T], ...]
```

The final part is a non-empty `Inline`. This is an application invariant: generation always has a contextual terminal prompt. Adjacent `Inline[str]` parts are normalized into one part before tokenization. ContextBlock boundaries are preserved, including adjacent ContextBlocks.

Separators and control text are explicit Inline content. There is no separator-specific part because its behavior is ordinary causal prefill.

Dataset prompts use one default composition: the preamble Inline is followed directly by the first ContextBlock; a single-space Inline separates consecutive ContextBlocks and precedes the terminal task prompt. Templates may add chat-control text to the leading or terminal Inline, but they preserve ContextBlock contents and order.

## Tokenization

A prompt is compiled once into:

```python
TokenizedPrompt(
    input_ids: Tensor,  # shape [sequence_length]
    parts: tuple[TokenSpan, ...],
)
```

`TokenSpan` records only the part kind and a half-open `[start, end)` range into `input_ids`. Spans form an ordered, gap-free partition of the canonical token stream. Per-part token IDs are views derived from `input_ids`, never a second authoritative copy.

Every normalized part is a deliberate token boundary. The compiler batch-tokenizes the part texts without padding or special tokens, then concatenates those results into the canonical IDs. This is necessary because a ContextBlock must remain independently preparable and ordinary BPE tokenization may otherwise create a token that crosses a semantic part boundary. Adjacent Inline text is merged first so normal prose does not acquire unnecessary boundaries.

No consumer re-tokenizes prompt text. Teacher, student, and eval all consume the compiler output, so the deliberate part boundaries cannot create teacher/student drift.

## Teacher And Generation Cache

The teacher consumes canonical flat token IDs and ignores part kinds. The student consumes the same IDs and part spans. Teacher forcing appends `teacher_sequence[:-1]` to the terminal Inline path, preserving the existing next-token shift.

Teacher generation cache identity is derived from the raw semantic sample: ordered document contents, raw query, actually rendered few-shot examples, and response-relevant task fields. Token IDs, tokenizer, prompt template, think markup, teacher model, and generation settings are deliberately excluded. This lets one semantic sample address teacher artifacts produced by different decoding experiments without treating rendering metadata as content.

Teacher generation follows the configured decoding policy. Sampling defaults to disabled; explicitly setting `do_sample=true` remains supported, and the sampled result is materialized on the first cache miss and then reused. `num_return_sequences` may be greater than one. Every returned sequence is stored under the same canonical prompt and becomes an independent teacher target; prompt and ContextBlock preparation are reused, while loss remains summed over every target token.

One standalone cache artifact represents one teacher-generation experiment. Its adjacent resolved config records how it was produced, while training uses only semantic sample coverage and operational CE/KL tensor requirements.

## Method Semantics

| Method family | Prompt consumption |
| --- | --- |
| `full_recompute` | The only lossless complete-prompt baseline. Run the canonical flat IDs as an ordinary causal prefill without prepared KV artifacts or re-tokenization. |
| `single_cache` | Cache everything before the terminal Inline, then run the terminal Inline. |
| `no_cache` | Remove every ContextBlock from the canonical prompt, retain all Inline tokens in order, and run the remaining canonical IDs as an ordinary causal prefill without prepared KV artifacts or re-tokenization. |
| `no_recompute`, `kvpacket`, `sempic`, `sempic_kvpacket` | Prepare each ContextBlock with the method environment; globally re-rotate prepared keys into the final interleaved layout, then process all compact Inline rows together at each decoder layer. |
| `cache_blend`, `a3`, `rand_recompute` | Selection and recompute ratios apply only to candidate ContextBlock tokens. Inline is ordinary causal prefill and is excluded from the candidate count. A first ContextBlock is an unchanged exact prefix only when it begins at logical position zero. |
| `epic` | Recompute the first configured number of tokens from every ContextBlock. Inline content remains ordinary contextual input and is not an EPIC budget unit; there is no first-block exception. |
| `sam_kv` | Every token-bearing part before the terminal Inline contributes to the peer statistic. Only ContextBlocks are document-selection units. Inline is normally prefetched in logical order, and the terminal Inline is excluded from the peer statistic. During final recomputation, only reused ContextBlock KV is fused; Inline and terminal KV are overwritten by normal prefill values. |
| `sink` | Retain the sink method's ContextBlock policy while normally prefilling Inline in logical order, including Inline between ContextBlocks. |

Methods may create runtime KV layouts keyed by prompt part index. Those layouts are runtime artifacts and are not fields of PromptSequence or TokenizedPrompt. Methods whose physical KV length differs from source token length own the source-to-physical mapping.

Eval methods are selected by registered method name. Adding a method requires registering its evaluator and preparation family; `run_eval` does not accept an unregistered evaluator callable as an alternate public preparation contract.

The terminal-query attention provider uses the same interleaved layout and compact Inline execution as evaluation. Its observable eager backend executes every Inline row needed for the real hidden-state path while streaming only the selected terminal query rows to reducers. It does not re-tokenize the prompt or run a second terminal-only attention computation.

## Student Prefill Execution

For a training microbatch:

1. Gather all ContextBlocks from all samples, pad them with an explicit validity mask, and prepare them in one batched model traversal with the configured LoRA/PacketWrapper environment.
2. Map prepared KVs back to `(sample_index, part_index)` without detaching gradients.
3. Build the final interleaved physical layout, globally re-rotate all prepared document keys once across layers and document occurrences, and run all contextual Inline tokens and teacher-forcing tokens in one batched layer-level traversal.

For current full-attention models, each compact Inline query receives a physical frontier derived from the shared layout. Visibility requires a valid key whose interleaved physical index does not exceed that frontier. Explicit logical positions remain authoritative for RoPE placement and future layer-aware cache policies. This prevents an early Inline from attending to a later ContextBlock while allowing later Inline tokens to attend to earlier Inline and ContextBlock tokens.

The production contract is batched ContextBlock preparation followed by one compact Inline decoder traversal, not one stock Hugging Face forward. ContextBlocks never enter the online query projection, attention-output, or MLP path. Prepared caches are immutable derived inputs; final interleaved K/V assembly and re-rotation are out of place.

## Invariants

- One prompt has one canonical token stream.
- The final part is a non-empty Inline.
- Inline text may appear before, between, and after ContextBlocks.
- ContextBlocks remain individually addressable.
- Teacher and student use the same next-token target IDs and target order. Method-owned soft/physical KV tokens may change the student's RoPE positions without changing that target alignment.
- Padding is represented by lengths or explicit validity masks, never inferred solely from `token_id != pad_token_id`.
- Document-key re-rotation occurs once after placement is known and never inside the decoder-layer loop.
- Train, eval, generation, and attention analysis consume one shared physical placement and visibility authority.
- Batching must preserve autograd, optimizer boundaries, and the current summed CE/KL loss semantics.
- Eval method policy remains method-owned; PromptSequence does not flatten algorithm differences into a generic interpreter.

## Acceptance Criteria

- Tokenization tests cover deliberate part boundaries, adjacent Inline normalization, multiple ContextBlocks, and an interleaved Inline.
- Training tests prove teacher/student token equality, correct teacher-forcing loss positions, and multi-target teacher branching.
- Student tests prove ContextBlocks from several samples are prepared in batches and terminal work is a real model batch.
- Every registered eval method is covered by the method matrix or produces a deliberate capability error for an unsupported prompt shape.
- Existing unified train/eval contract tests and Python compilation pass.
