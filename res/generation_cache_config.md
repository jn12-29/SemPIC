# Generation Cache Configuration Reference

`run_generation_cache.py` creates one immutable teacher-generation experiment. It accepts exactly one JSON config and writes exactly three files under a new `output_dir`:

```text
<output_dir>/
├── cache.safetensors
├── manifest.json
└── resolved_config.json
```

The command refuses an existing output path. Generated tensor payloads are appended once directly to a private temporary `cache.safetensors`. Finalization backfills its reserved safetensors header and writes `manifest.json` without copying or rewriting the tensor payload. The completed three-file private directory is then published with one rename. There are no per-entry chunk files and no append, resume, merge, or overwrite mode. If the reserved header capacity is insufficient, generation fails and removes the private temporary artifact without publishing `output_dir`.

The runtime format is safetensors-only.

```bash
CUDA_VISIBLE_DEVICES=0 uv run python run_generation_cache.py \
  generation_cache_config/qwen_3_8b_teacher/biography_greedy_kl.json
```

## Fields

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `model` | object | yes | Teacher model and decoding configuration. |
| `data_configs` | list[object] | yes | Non-empty dataset list using the same fields as training. |
| `store_logits` | bool | yes | Store logits for KL training when `true`; store sequences only when `false`. |
| `gen_batch_size` | int | no | Teacher generation batch size. Defaults to `1`. |
| `cache_device` | str | no | Streaming cache device. It must be `"cpu"` and defaults to `"cpu"`. Only the current generated entry is staged in CPU memory before its tensor bytes are appended; entry metadata is retained until finalization. |
| `seed` | int | no | Python and PyTorch seed. Defaults to `42`. |
| `output_dir` | str | yes | New artifact directory. It must not already exist, including as a dangling symlink. |

`model` contains `model_path`, optional `tokenizer_path` (defaults to `model_path`), optional `dtype` (defaults to `"bfloat16"`), optional `device` (defaults to `"cuda:0"`), and optional `generation_kwargs`. EOS behavior comes from the teacher model's generation config unless explicitly overridden; `stop_strings` is only needed for an additional text-level stopping rule.

`resolved_config.json` records the effective teacher generation settings and tokenizer state. It is provenance only: training never reads it as a compatibility gate.

Dataset-specific defaults are the same as training. In particular, HotpotQA `data_kwargs.difficulty` defaults to `["hard"]`; an explicit value must be a non-empty list containing only `"easy"`, `"medium"`, and `"hard"`. A cache must be regenerated whenever its effective dataset filtering changes, because training validates required semantic sample keys rather than the adjacent provenance file.

## Identity And Training Use

Each cache key hashes the raw semantic sample: ordered document contents, raw query, actually rendered few-shot examples, and response-relevant task fields. Token IDs, tokenizer, prompt template, think markup, teacher model, and generation settings are not key fields.

Point training at the generated artifact directory:

```json
"cache_path": "./generation_cache/artifacts/qwen_3_8b_teacher/biography_greedy_kl"
```

Training treats the artifact as immutable. It reads `manifest.json` for coverage and tensor-shape checks, then maps only the requested entry from `cache.safetensors`. It does not preload the payload into CPU memory. The cache teacher, tokenizer path and current tokenizer state (`padding_side`, pad ID, and EOS ID), and dtype must match training. Multiple compatible artifact directories can be listed for mixture training; their logit-storage mode must also match, while per-domain generation lengths may differ. Conflicting duplicate semantic keys fail before training.
