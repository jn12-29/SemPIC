# AGENTS.md

Repository-specific instructions for agents working in this project.

## Python Environment

- Use `uv` for the project environment. Create it in the repository root:
  ```bash
  uv venv --python 3.12 .venv
  ```
- Install or refresh the project environment from `pyproject.toml` and `uv.lock`:
  ```bash
  uv sync
  ```
- Run project commands through uv so they use the repository `.venv`:
  ```bash
  uv run python -m unittest tests.test_lora_cache
  uv run python run_eval.py <config.json> --overwrite
  ```
- `pyproject.toml` and `uv.lock` are the uv-owned dependency source.
- For ad hoc package installs into the local environment, prefer `uv pip` from the repository root. If the current working directory or active environment is ambiguous, target the environment explicitly:
  ```bash
  uv pip install --python .venv/bin/python <package>
  ```
- Do not use `.venv/bin/python -m pip`; this uv-created environment may not include `pip`.

## Entrypoints And Config Paths

- Unified training uses `run_train.py` with configs under `train_config/`. `output_dir` is the method root `train_outputs/<model>/<dataset>/<method>/`, where method is `kvpacket`, `sempic`, or `joint`; each concrete run is a timestamp child directory.
- Evaluation uses `run_eval.py` with configs under `eval_config/`. Each call writes an invocation under `eval_outputs/_invocations/`; every config that starts evaluation writes a timestamped run under `eval_outputs/<config-scope>/<eval-method>/` and atomically publishes its canonical result to adjacent `eval_results/` after success.
- Evaluation visualization uses `run_eval_dashboard.py`; scan `eval_config` for latest stable results or `eval_outputs` for timestamped history, but do not scan both in one process because they contain duplicate result copies.
- Attention reprocessing uses `run_process_attention.py --run-dir <run> [--suffix <name>]`; every call creates a new timestamped directory under `<run>/analysis/` and preserves sibling variants.
- Deep field references live in `res/train_config.md`, `res/eval_config.md`, and `res/attention_analysis.md`; update those when changing their contracts.

## Hugging Face Access

- If the official Hugging Face endpoint is unreachable, or the machine is running in mainland China without direct external network access, enable the mirror before model or dataset downloads:
  ```bash
  source export_hf_mirror.sh
  ```
- The mirror script exports `HF_ENDPOINT=https://hf-mirror.com`; keep using that environment for Hugging Face dataset downloads as well as model/tokenizer downloads.

## GPU Selection

- Before every training, evaluation, or other CUDA model-loading run, check idle GPUs first:
  ```bash
  nvidia-smi
  ```
- Use `CUDA_VISIBLE_DEVICES=<physical_gpu_id>` to bind the run to an idle GPU:
  ```bash
  CUDA_VISIBLE_DEVICES=0 uv run python run_train.py <config.json>
  CUDA_VISIBLE_DEVICES=0 uv run python run_eval.py <config.json> --overwrite
  ```
- Most configs use `model.device: "cuda:0"`; with `CUDA_VISIBLE_DEVICES` set, `cuda:0` refers to the first visible GPU, not necessarily physical GPU 0.

## Config Rules

- `_default.json` files fill missing fields in sibling configs. Explicit fields in the selected config win, including nested fields such as `model.model_path`.
- `train.targets` may include only enabled components from `lora.enabled` and `packet_wrapper.enabled`.
- `loss.type == "kl"` stores teacher logits in the generation cache; `loss.type == "ce"` stores teacher sequences only.
- A concrete training run is `<output_dir>/<YYYYMMDD_HHMMSS[_suffix]>/`. LoRA is saved to `<run_dir>/lora/`, PacketWrapper to `<run_dir>/packet_wrapper.pt`, and joint training writes both into the same run directory.
- `lora.adapter_name` must be `default` so the adapter is saved directly in `<run_dir>/lora/`.
- Config `run_suffix` is optional; CLI `--run-suffix` overrides it for new runs. It must start with an alphanumeric character and may otherwise contain letters, digits, `.`, `_`, and `-`. Resume exactly one concrete run with `--resume-from <run_dir>`; symlinks are rejected, and `--run-suffix` cannot be combined with resume.
- Training replaces `<run_dir>/checkpoint.pt` after every complete epoch and removes it only after final artifacts save successfully. Training does not create a moving run alias. Evaluation artifact paths are repository-root-relative and pin concrete runs; config stems include unique training-run suffixes so stable results remain checkpoint-specific.
- Eval top-level `run_suffix` is optional. CLI `--run-suffix` overrides it for config runs and is the only suffix used by the invocation directory.
- Evaluation writes the canonical result in the timestamped config run first, then atomically copies real JSON to `<config-parent>/eval_results/<config-stem>_result.json`. `--overwrite` controls whether a new run starts when the stable result exists; failures do not replace the prior stable result.
- Eval method names are explicit: `kvpacket` uses `packet_wrapper.path`, `sempic` uses `lora.path`, and `sempic_kvpacket` requires both.

## Targeted Verification

- Evaluation dashboard loader and aggregation tests:
  ```bash
  uv run python -m unittest tests.test_eval_dashboard
  ```
- Unified training/eval contract tests:
  ```bash
  uv run python -m unittest tests.test_lora_cache tests.test_train_filler tests.test_runtime
  ```
- Attention run, variant, and processing tests:
  ```bash
  uv run python -m unittest tests.test_attention_run tests.test_attention_variants tests.test_attention_processing
  ```
- Syntax-check changed Python entrypoints and package files:
  ```bash
  uv run python -m compileall -q run_train.py run_eval.py run_generation_cache.py run_eval_dashboard.py run_build_packet.py run_attention_analysis.py run_process_attention.py plot_scripts sempic tests
  ```
- Validate changed JSON configs before finishing:
  ```bash
  uv run python -m json.tool <config.json> >/tmp/config.jsoncheck
  ```
- For eval config changes, load through the real inheritance and validation path:
  ```bash
  uv run python - <<'PY'
  from sempic.utils.config import load_config_file
  from run_eval import load_eval_config
  load_eval_config(load_config_file("<config.json>", default_config_file="_default.json"))
  PY
  ```
