# Evaluation Configuration Reference

`run_eval.py` evaluates the cache combination methods listed below. The artifact-backed method names are `kvpacket`, `sempic`, and `sempic_kvpacket`.

Methods are selected only by registered method name. A new method must register both its evaluator and preparation family; arbitrary unregistered evaluator callables are not a supported entry point.

## Top-Level Fields

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `model` | dict | yes | Base causal LM configuration. |
| `dataset` | dict | yes | Evaluation dataset configuration. |
| `cache_comb` | dict | yes | Cache combination method and kwargs. |
| `packet_wrapper` | dict | method-dependent | PacketWrapper artifact path. |
| `lora` | dict | method-dependent | LoRA artifact path for document KV construction. |
| `compress` | dict \| null | no | Optional KV compression settings. |
| `quantization` | dict \| null | no | Optional KV quantization settings. |
| `seed` | int | yes | Dataset seed. |
| `run_suffix` | str \| null | no | Optional readable suffix for a timestamped config run. CLI `--run-suffix` takes precedence. |
| `logging` | dict | no | Runtime log level settings. |
| `debug_dump` | dict | no | Runtime debug artifact settings. |

## Method Names

| Method | Artifact contract | Description |
| --- | --- | --- |
| `no_cache` | none | Remove every ContextBlock from the canonical prompt, retain all Inline tokens, and prefill the remaining canonical IDs normally without re-tokenization. |
| `full_recompute` | none | The lossless full-prompt baseline: prefill the complete canonical IDs normally without prepared KV artifacts or re-tokenization. |
| `no_recompute` | none | Concatenate document KV without repair. |
| `cache_blend` | none | CacheBlend over candidate ContextBlock tokens; Inline is normal causal prefill and is excluded from its ratio. |
| `epic` | none | EPIC baseline. |
| `sink` | none | Attention sink baseline with Inline normally prefetched in logical order. |
| `rand_recompute` | none | Random recomputation over candidate ContextBlock tokens; Inline is excluded from selection and ratio. |
| `kvpacket` | `packet_wrapper.path` | PacketWrapper header/trailer method. |
| `sempic` | `lora.path` | SemPIC document-KV adapter method. |
| `sempic_kvpacket` | `packet_wrapper.path` and `lora.path` | Combine SemPIC and PacketWrapper artifacts, from either joint or independent training. |
| `sam_kv` | none | SAM-KV selects ContextBlock document units and normally prefills Inline in logical order. |
| `a3` | none | A3 over candidate ContextBlock tokens; Inline is normal causal prefill and is excluded from its ratio. |
| `single_cache` | none | Single combined document cache baseline. |

`packet_wrapper.path` and `lora.path` are rejected for methods that do not list them above.

`full_recompute` is the only lossless baseline that performs ordinary prefill over the complete canonical prompt. It does not consume prepared KV artifacts.

## Method Kwargs

| Method | Kwarg | Contract |
| --- | --- | --- |
| `a3` | `recompute_ratio` | Float in `(0, 1]`. `int(ratio * candidate_tokens)` must select at least one ContextBlock token; use `no_recompute` when the effective selection is empty. |
| `cache_blend` | `recompute_ratio` | Float in `(0, 1]`. `int(ratio * candidate_tokens)` must select at least one ContextBlock token; use `no_recompute` when the effective selection is empty. |
| `epic` | `recompute_tokens` | Non-negative integer number of leading tokens to recompute independently in every ContextBlock. Inline tokens are excluded. |
| `rand_recompute` | `recompute_ratio` | Float in `[0, 1]` selecting a random fraction of candidate ContextBlock tokens. Inline tokens are excluded. |
| `rand_recompute` | `seed` | Optional random seed for repeatable token selection. |
| `sam_kv` | `stable_layers` | Non-empty list of zero-based decoder layer indices used to estimate document-block relevance. Every index must be valid for the selected model. |
| `sam_kv` | `num_initial_tokens` | Non-negative integer number of leading tokens retained from every ContextBlock. |
| `sam_kv` | `num_local_tokens` | Positive integer number of trailing tokens retained from every ContextBlock and used to construct peer query states. |
| `sam_kv` | `block_size` | Positive integer number of interior ContextBlock tokens represented by each relevance-scoring block. |
| `sam_kv` | `fuse_theta` | Float interpolation weight used when fusing recomputed KV into the prepared ContextBlock KV. |

## Artifacts

`packet_wrapper.path` points to a saved PacketWrapper artifact:

```json
"packet_wrapper": {
    "path": "./train_outputs/qwen_3_4b/biography/kvpacket/<run-directory>/packet_wrapper.pt"
}
```

`lora.path` points to a saved PEFT adapter subdirectory:

```json
"lora": {
    "path": "./train_outputs/qwen_3_4b/biography/sempic/<run-directory>/lora"
}
```

During eval, LoRA is enabled only while producing document KV from document `inputs_embeds`; it is disabled before the query and generation path.

Artifact-backed evaluation uses the final training layout:

| Evaluation method | Training method directory | Paths |
| --- | --- | --- |
| `kvpacket` | `kvpacket` | `packet_wrapper.path = <concrete_run_dir>/packet_wrapper.pt` |
| `sempic` | `sempic` | `lora.path = <concrete_run_dir>/lora` |
| `sempic_kvpacket` | `joint` for joint training | Joint-trained configs use both paths from one concrete joint run; independent combinations pin one concrete KVPacket run and one concrete SemPIC run. |

Artifact paths must be repository-root-relative strings beginning with `./`. Runtime validation rejects absolute paths, parent traversal, and any `latest` path segment; it intentionally does not require the artifact to exist while parsing a config. Repository-maintained configs select artifacts from concrete training runs and never follow a moving training alias.

Create an artifact-backed eval config after its concrete training run exists. Place it beside the dataset's `_default.json` so missing model and dataset fields are inherited while fields in the concrete config win. Use a unique, recognizable training-run suffix in the config stem; the standard form is `<eval-method>__<training-run-directory>`. Cross-domain configs use `<eval-method>__<source-dataset>__<training-run-directory>` so runs with the same directory name under different dataset scopes remain distinct. Stable result names are derived from config stems, producing one unambiguous result per checkpoint within each config directory. External configs may use other stems, but each stem still owns one stable result identity.

## Run Directories And Result Publication

Every `run_eval.py` call allocates `eval_outputs/_invocations/<YYYYMMDD_HHMMSS[_suffix]>/` with `run.log` and `cli_args.json`. The invocation log owns config discovery, warnings, skipped configs, and overall status. Its optional suffix comes only from CLI `--run-suffix`.

After a config loads and validates, and immediately before evaluation resources are created, it allocates:

```text
eval_outputs/<config-scope>/<eval-method>/<YYYYMMDD_HHMMSS>_<config-stem>[_suffix]/
```

For configs under `eval_config/`, `<config-scope>` is the config parent relative to `eval_config/`, such as `qwen_3_8b/biography` or `llama_3_1_8b/cross_domain/biography`. An external config uses `_external/<safe-parent-name>/<safe-config-stem>`. `<eval-method>` is `cache_comb.method`. A skipped config has no config-run directory.

Top-level `run_suffix` applies to the config run; CLI `--run-suffix` overrides it. A suffix must start with an alphanumeric character and may otherwise contain letters, digits, dots, underscores, and hyphens. Timestamp collisions append `_1`, `_2`, and so on.

The config-run directory contains `run.log`, `eval_config.json`, `resolved_config.json`, `cli_args.json`, the canonical `<config-stem>_result.json`, and a lazily created `debug/`. The canonical result is written first. Evaluation then atomically copies real JSON to `<config-parent>/eval_results/<config-stem>_result.json`; an external config publishes beside its own parent directory. Both copies preserve the configured concrete repository-relative artifact paths.

`--overwrite` controls whether a new run starts when the stable adjacent result already exists; it never overwrites a timestamped run. A failed config run remains available for diagnosis but does not publish or alter the stable result. If stable publication fails, the command fails, the prior stable result remains intact, and the canonical result remains in the config run.

For the dashboard, scan `eval_config` to view latest stable results or `eval_outputs` to view timestamped history. Do not scan both roots in the same dashboard process because completed results exist in both locations.

## Result Metrics

Document KV construction, PacketWrapper preparation, compression, quantization, and an exact-shape synthetic attention compilation warmup are completed before each sample's TTFT timer starts. The warmup does not execute the request's layout, KV re-rotation, decoder traversal, cache assembly, or lm-head work. The timer covers the formal online layout, KV re-rotation, compact prefill, decode-cache construction, generation setup, and first generated token, with device synchronization at both boundaries.

`ttft` is the mean of the post-warmup sample measurements and equals `ttft_mean`. Results also report `ttft_p50`, `ttft_p90`, `ttft_p99`, `ttft_min`, `ttft_max`, population `ttft_std`, and `ttft_count`. Percentiles use linear interpolation between ordered sample values. No dataset sample is skipped as a warmup sample.

## `logging`

`logging.level` optionally sets the runtime log level and defaults to `"INFO"`; CLI `--log-level` takes precedence. Run placement and `run.log` naming are fixed by the run-directory contract.

## `model`

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `model_path` | str | yes | HuggingFace model id or local model directory. |
| `dtype` | str | no | Model dtype. Defaults to `"float32"`. |
| `device` | str | no | Device. Defaults to `"cuda:0"`. |
| `generation_kwargs` | dict | no | Generation settings passed to `GenerationConfig`. |

## `dataset`

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `dataset_name` | str | yes | Dataset key. |
| `num_samples` | int | yes | Number of eval samples. |
| `num_data_strs` | int | yes | Number of retrieved document strings per sample. |
| `num_shots` | int | yes | Number of in-context examples. |
| `subset` | str | yes | Dataset subset. |
| `split` | str | yes | Dataset split. |
| `seed` | int | yes | Dataset seed. |
| `data_kwargs` | dict | no | Dataset-specific settings. |
| `template` | str | no | Prompt template key. Defaults to `"default"`. |
| `template_kwargs` | dict | no | Template-specific settings. |

For HotpotQA, `data_kwargs.difficulty` defaults to `["hard"]`. An explicit value must be a non-empty list containing only `"easy"`, `"medium"`, and `"hard"`. The filter applies to both few-shot and evaluation samples after the full split is shuffled by `seed`.

## Examples

`kvpacket`:

```json
{
    "cache_comb": {"method": "kvpacket", "kwargs": {}},
    "packet_wrapper": {"path": "./train_outputs/qwen_3_4b/biography/kvpacket/<run-directory>/packet_wrapper.pt"}
}
```

`sempic`:

```json
{
    "cache_comb": {"method": "sempic", "kwargs": {}},
    "lora": {"path": "./train_outputs/qwen_3_4b/biography/sempic/<run-directory>/lora"}
}
```

`sempic_kvpacket`:

```json
{
    "cache_comb": {"method": "sempic_kvpacket", "kwargs": {}},
    "packet_wrapper": {"path": "./train_outputs/qwen_3_4b/biography/joint/<run-directory>/packet_wrapper.pt"},
    "lora": {"path": "./train_outputs/qwen_3_4b/biography/joint/<run-directory>/lora"}
}
```

Run a file or directory:

```bash
uv run python run_eval.py eval_config/qwen_3_4b/biography/<config-name>.json --overwrite
uv run python run_eval.py eval_config/qwen_3_4b/biography/ --overwrite
```

For NIAH, `data_kwargs.max_len` is applied to the tokenized context length when the built-in loader passes a tokenizer; this keeps train and eval sample filtering aligned.
