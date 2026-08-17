# Training Configuration Reference

`run_train.py` is the public training entrypoint. It supports Packet-only, LoRA-only, and joint Packet+LoRA training through explicit component switches and training targets.

## Top-Level Fields

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `total_epoch` | int | yes | Number of training epochs. |
| `batch_size` | int | yes | Number of samples per optimizer step. All target optimizers step on this same boundary. |
| `forward_batch_size` | int | no | Number of samples processed together by one student forward/backward microbatch. Defaults to `1`; must divide `batch_size`. SDPA and FlexAttention support larger ragged microbatches subject to available GPU memory. |
| `attention_backend` | string | no | Training attention preference: `auto` (default) or `flex` tries FlexAttention on CUDA and falls back to SDPA only if its integrated loss/backward preflight fails; `sdpa` selects SDPA directly. `run_train.py --attention-backend` overrides this field. |
| `gen_batch_size` | int | yes | Batch size for fallback online teacher generation when `cache_path` is null. |
| `model` | dict | yes | Base causal LM configuration. |
| `cache_device` | str | no | Backing device for configured artifacts or online cache tensors. Configured artifacts require `"cpu"`; online generation defaults to `"cuda:0"`. |
| `cache_path` | str \| list[str] \| null | no | One immutable safetensors artifact directory, a non-empty list composed in order, or null for fallback online generation. Missing, duplicate, incompatible, or conflicting paths fail before training. |
| `output_dir` | str | yes | Method root. Use `train_outputs/<model>/<dataset>/<method>`. |
| `run_suffix` | str \| null | no | Optional readable suffix for a new timestamp run. CLI `--run-suffix` takes precedence. |
| `kv_gradient_checkpointing` | bool | no | Recomputes document KV forwards during backward to reduce activation memory while preserving `use_cache=True`. Defaults to `true`. |
| `seed` | int | no | Shuffle seed. Defaults to `42`. |
| `train` | dict | no | Training target selection. |
| `loss` | dict | no | Shared student loss configuration. |
| `lora` | dict | yes | LoRA component configuration. |
| `packet_wrapper` | dict | yes | PacketWrapper component configuration. |
| `optimizers` | dict | yes | Optimizer and scheduler configs keyed by training target. |
| `data_configs` | list[dict] | yes | Training datasets. |
| `logging` | dict | no | Runtime log level settings. |
| `debug_dump` | dict | no | Runtime debug artifact settings. |

## `model`

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `model_path` | str | yes | HuggingFace model id or local model directory. |
| `dtype` | str | no | Model dtype passed to `from_pretrained`; defaults to `"bfloat16"`. |
| `device` | str | no | Device or `"auto"`; defaults to `"cuda:0"`. |
| `generation_kwargs` | dict | no | Fallback teacher settings used only when `cache_path` is null. They are ignored when an artifact is configured. |

## Components And Targets

`lora.enabled` and `packet_wrapper.enabled` decide which components participate in document KV construction. `train.targets` decides which enabled components are optimized and saved.

If `train.targets` is omitted, it defaults to all enabled components. A target must be enabled, and at least one component must be enabled.

```json
"train": {
    "targets": ["lora", "packet_wrapper"]
}
```

The base model is always frozen. Enabled components that are not listed in `train.targets` participate in the forward path but do not enter an optimizer. Only trained targets produce final artifacts:

| Targets | Method directory | Final artifacts |
| --- | --- | --- |
| `packet_wrapper` | `kvpacket` | `<run_dir>/packet_wrapper.pt` |
| `lora` | `sempic` | `<run_dir>/lora/` |
| `lora`, `packet_wrapper` | `joint` | `<run_dir>/lora/`, `<run_dir>/packet_wrapper.pt` |

For example, joint Qwen3-8B Biography training uses `output_dir: "./train_outputs/qwen_3_8b/biography/joint"`.

## `loss`

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `"kl"` \| `"ce"` | no | Defaults to `"kl"`. |
| `tau` | float | no | KL temperature. Used only when `type` is `"kl"`; defaults to `1.0`. |

KL training requires one floating-point logit tensor per cached teacher sequence. CE training requires teacher sequences only.

Training never writes a configured cache artifact. It validates coverage and KL availability from `manifest.json` without loading tensor payloads, and loads only each requested sample from `cache.safetensors`. The artifact teacher, tokenizer path and current tokenizer state (`padding_side`, pad ID, and EOS ID), and dtype must match training. When `cache_path` is a list, training composes the union for mixture training. The artifacts must also agree on whether logits are stored; their decoding limits may differ. Identical duplicate entries are accepted, conflicting duplicates fail. `resolved_config.json` remains provenance rather than a training-time gate. Cache lookup uses raw semantic sample identity rather than token IDs, templates, tokenizer settings, or generation settings. See [generation_cache_config.md](./generation_cache_config.md) for artifact generation.

## `lora`

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `enabled` | bool | yes | Enables LoRA for document KV construction. |
| `rank` | int | no | PEFT LoRA rank. Defaults to `8`. |
| `alpha` | int | no | PEFT `lora_alpha`. Defaults to `16`. |
| `dropout` | float | no | PEFT `lora_dropout`. Defaults to `0.0`. |
| `target_modules` | list[str] | no | Module names used by PEFT. Defaults to Q/K/V/O projections. |
| `adapter_name` | str | no | Runtime PEFT adapter name. It must be `"default"` so the adapter is saved directly in the fixed `lora/` directory. |
| `init_path` | str \| null | no | Existing PEFT adapter directory to initialize from. |

When `"lora"` is trained, the saved artifact is `<run_dir>/lora`. Evaluation pins that concrete run with a repository-relative path.

## `packet_wrapper`

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `enabled` | bool | yes | Enables PacketWrapper document embedding wrapping. |
| `header_len` | int \| null | init-dependent | Header soft-token count. Required when `init_path` is null. |
| `trailer_len` | int \| null | init-dependent | Trailer soft-token count. Required when `init_path` is null. |
| `dtype` | str \| null | no | Wrapper tensor dtype. Defaults to `model.dtype`. |
| `init_path` | str \| null | no | Existing `.pt` PacketWrapper artifact to initialize from. |

When `"packet_wrapper"` is trained, the saved artifact is `<run_dir>/packet_wrapper.pt`. Evaluation pins that concrete run with a repository-relative path. Joint training writes this file and `lora/` into the same run directory.

## Run Directories, Checkpoint, And Resume

A new run allocates `<output_dir>/<YYYYMMDD_HHMMSS[_suffix]>/`. `run_suffix` must start with an alphanumeric character and may contain letters, digits, dots, underscores, and hyphens. `--run-suffix` overrides the config value. Timestamp collisions append `_1`, `_2`, and so on.

The run directory contains `run.log`, `resolved_config.json`, `train_config.json`, `cli_args.json`, the current `checkpoint.pt`, and final target artifacts. The `debug/` directory is created lazily when debug artifacts are emitted. Training replaces `<run_dir>/checkpoint.pt` after every complete epoch. The checkpoint contains the completed-epoch position and the trained component, optimizer, scheduler, and random-number-generator state needed to continue training.

Resume one interrupted run with `--resume-from <run_dir>`. The path must be a concrete direct child of the configured `output_dir`; symlinks are rejected. Resume accepts exactly one training config and cannot be combined with `--run-suffix`. It reuses the same run directory.

After every configured epoch completes and final target artifacts save successfully, training deletes `checkpoint.pt`. If final artifact saving fails, the checkpoint remains available for resume. Training does not create a moving run alias. A configured teacher generation cache remains immutable and outside the run directory; a fallback online cache exists only in memory.

## `logging`

`logging.level` optionally sets the runtime log level and defaults to `"INFO"`; CLI `--log-level` takes precedence. Run placement and `run.log` naming are fixed by the run-directory contract.

## `optimizers`

Each training target has its own optimizer and scheduler config:

```json
"optimizers": {
    "lora": {
        "opt_config": {"lr": 5e-4, "weight_decay": 0.0},
        "scheduler_config": {"start_factor": 1.0, "end_factor": 0.0}
    },
    "packet_wrapper": {
        "opt_config": {"lr": 5e-4, "weight_decay": 0.0},
        "scheduler_config": {"start_factor": 1.0, "end_factor": 0.0}
    }
}
```

`opt_config` is passed to `torch.optim.AdamW`. `scheduler_config` is passed to `torch.optim.lr_scheduler.LinearLR`; if `total_iters` is `0` or omitted, it is set from the configured sample count, `batch_size`, and `total_epoch`.

## Training Data

Each entry in `data_configs` is passed to `get_ret_eval_generator`:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `dataset_name` | str | yes | Dataset key. |
| `num_samples` | int | yes | Number of samples from this dataset config. |
| `num_data_strs` | int | yes | Number of retrieved document strings per sample. |
| `num_shots` | int | yes | Number of in-context examples. |
| `subset` | str | yes | Dataset subset. |
| `split` | str | no | Defaults to `"train"`. |
| `seed` | int | no | Defaults to `42`. |
| `data_kwargs` | dict | no | Dataset-specific settings. |
| `template` | str | no | Prompt template key. |
| `template_kwargs` | dict | no | Template-specific settings. |

For NIAH, `data_kwargs.max_len` is applied to the tokenized context length when the built-in loader passes a tokenizer; this is used to filter out the longest contexts before training.

For HotpotQA, `data_kwargs.difficulty` defaults to `["hard"]`. An explicit value must be a non-empty list containing only `"easy"`, `"medium"`, and `"hard"`. The filter applies to both few-shot and training samples after the full split is shuffled by `seed`.

## Examples

Representative configs live under `train_config/qwen_3_4b/biography/`:

| File | Components | Targets | Loss |
| --- | --- | --- | --- |
| `packet_only_kl.json` | PacketWrapper | PacketWrapper | KL |
| `lora_only_kl.json` | LoRA | LoRA | KL |
| `packet_lora_joint_kl.json` | PacketWrapper + LoRA | PacketWrapper + LoRA | KL |

Run a file or directory:

```bash
uv run python run_train.py train_config/qwen_3_4b/biography/packet_only_kl.json
uv run python run_train.py train_config/qwen_3_4b/biography/
```
