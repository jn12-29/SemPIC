<div align="center">

# SemPIC: Learning Semantic Position-Independent KV Caches

Hui Xie · Peng Xiao · Yutong Deng · Shuoran Dou · Jian Yang · Jinyang Guo

Beihang University

[![arXiv](https://img.shields.io/badge/arXiv-2607.28069-b31b1b.svg)](https://arxiv.org/abs/2607.28069)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg)](https://www.python.org/)

**Compile reusable native document KVs offline; serve them with an unchanged pretrained Reader.**

</div>

SemPIC learns semantic position-independent KV caches for multi-document RAG. A LoRA-enabled **Writer** compiles each reusable document into native per-layer KVs through behavioral distillation, while the pretrained **Reader** and its online cache-hit decoding path remain unchanged.

![SemPIC trains a document Writer through the native KV interface and reuses the compiled caches with an unchanged Reader.](./assets/sempic_overview.png)

## Core Idea

Position-independent caching (PIC) prepares each reusable document once, relocates its keys to the request-time positions, and links it with other cached documents. RoPE re-rotation fixes positional phase, but independently compiled KVs still lack the future context in which they will be consumed.

SemPIC addresses this contextual mismatch at the document representation itself:

- **Writer:** LoRA is enabled only while each document is compiled independently into native per-layer KVs.
- **Reader:** the same pretrained decoder consumes the linked caches with LoRA disabled.
- **Training:** self-supervised KL distillation matches the Reader's predictions to the base model's full-context behavior.
- **Serving:** compilation is paid offline; cache hits use standard KVs without document recomputation or an auxiliary online model.
- **Memory:** KV Gradient Checkpointing preserves the differentiable Writer–Reader cache interface while reconstructing Writer activations during backward.

![Position-independent cache designs address contextual mismatch through online recomputation, boundary adaptation, auxiliary interfaces, or SemPIC's native document Writer.](./assets/pic_overview.png)

## Key Results

The paper evaluates Llama-3.1-8B-Instruct, Qwen3-4B-Instruct-2507, and Qwen3-8B on Biography, HotpotQA, MuSiQue, and NIAH. Each model–task–method cell contains 100 examples, and quality is measured with corpus token micro-F1.

| Method                         | Overall mean micro-F1 |
| ------------------------------ | --------------------: |
| Full Recompute                 |                  0.62 |
| No Recompute                   |                  0.17 |
| KV Packet                      |                  0.53 |
| **SemPIC**                     |              **0.60** |
| **SemPIC + KV Packet (Joint)** |              **0.61** |

- SemPIC outperforms KV Packet in 10 of 12 model–task settings and raises its overall mean micro-F1 from 0.53 to 0.60.
- Joint training reaches 0.61, suggesting that document-wide and boundary adaptation can be complementary.
- SemPIC lowers Full-relative interior attention error compared with both No Recompute and KV Packet in all 12 settings under the paper's diagnostic metric.
- On Qwen3-8B, KV Gradient Checkpointing enables HotpotQA, MuSiQue, and NIAH training runs that otherwise run out of memory on one A100 80GB GPU.

These results compare cache-hit online computation after reusable KVs and learned wrappers have been prepared. See the [paper](https://arxiv.org/abs/2607.28069) for complete task-level results, metric boundaries, and limitations.

## Quick Start

### 1. Install

Create the Python 3.12 environment from the repository root with [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12 .venv
uv sync
```

The example configs use Hugging Face model identifiers. Before running a command, you may instead set `model.model_path` and any explicit `model.tokenizer_path` in the selected config or its sibling `_default.json` to the same local model directory. Explicit fields in a selected config override inherited defaults.

### 2. Run an Artifact-Free Baseline

This verifies the model, dataset, prompt, and evaluation path without requiring a trained adapter:

```bash
uv run python run_eval.py \
  eval_config/qwen_3_4b/biography/full_recompute.json --overwrite
```

You can replace `full_recompute.json` with `no_recompute.json` to evaluate direct reuse of independently prepared KVs.

### 3. Train SemPIC

Build the immutable teacher cache selected by the Qwen3-4B Biography training defaults, then train the LoRA Writer:

```bash
uv run python run_generation_cache.py \
  generation_cache_config/qwen_3_4b_teacher/biography_greedy_kl_1024.json

uv run python run_train.py \
  train_config/qwen_3_4b/biography/lora_only_kl.json
```

The teacher-cache directory contains `cache.safetensors`, `manifest.json`, and `resolved_config.json`. A successful training run saves its LoRA adapter under:

```text
train_outputs/qwen_3_4b/biography/sempic/<run-directory>/lora/
```

See the [generation-cache](./res/generation_cache_config.md) and [training](./res/train_config.md) references for dataset, optimization, checkpoint, and resume settings.

### 4. Evaluate the Trained Writer

Create an evaluation config beside `eval_config/qwen_3_4b/biography/_default.json` and pin the concrete training run:

```json
{
  "cache_comb": { "method": "sempic", "kwargs": {} },
  "lora": {
    "path": "./train_outputs/qwen_3_4b/biography/sempic/<run-directory>/lora"
  }
}
```

Then run:

```bash
uv run python run_eval.py \
  eval_config/qwen_3_4b/biography/sempic__<run-directory>.json --overwrite
```

Completed evaluations publish a stable result next to the source config under `eval_results/` and preserve timestamped run history under `eval_outputs/`. See the [evaluation reference](./res/eval_config.md) for the full method and result contract.

## Supported Methods

`cache_comb.method` selects the cache construction or linking strategy:

| Method            | Strategy                                            | Required trained artifact |
| ----------------- | --------------------------------------------------- | ------------------------- |
| `full_recompute`  | Recompute the complete prompt online                | None                      |
| `no_recompute`    | Directly link independently prepared document KVs   | None                      |
| `kvpacket`        | Add learned Header and Trailer boundary states      | `packet_wrapper.path`     |
| `sempic`          | Use native document KVs compiled by the LoRA Writer | `lora.path`               |
| `sempic_kvpacket` | Combine the SemPIC Writer with KV Packet boundaries | Both artifacts            |

`run_train.py` supports LoRA-only SemPIC training, PacketWrapper-only KV Packet training, and joint training. Joint runs save both artifacts in one timestamped directory.

## Advanced Workflows

### Joint Training

```bash
uv run python run_train.py \
  train_config/qwen_3_4b/biography/packet_lora_joint_kl.json
```

### Attention Analysis

Collect real-forward statistics and create the initial analysis variant:

```bash
uv run python run_attention_analysis.py \
  eval_config/qwen_3_4b/biography/full_recompute.json \
  eval_config/qwen_3_4b/biography/no_recompute.json \
  --analysis-config attention_config/paper_attention.json \
  --processing-config attention_config/processing_default.json \
  --max-samples 10 \
  --cpu-threads 4 \
  --run-name smoke
```

Add concrete artifact-backed SemPIC or KV Packet evaluation configs to compare learned methods. Existing statistics can be reprocessed without loading a model; see the [attention-analysis reference](./res/attention_analysis.md).

### Evaluation Dashboard

Browse either the latest stable results or timestamped history in one process:

```bash
# Latest stable results
uv run streamlit run run_eval_dashboard.py --server.headless true -- --root eval_config

# Timestamped history
uv run streamlit run run_eval_dashboard.py --server.headless true -- --root eval_outputs
```

The dashboard provides exact-run browsing, within-dataset comparison, cross-dataset views, and provenance inspection. It binds to `127.0.0.1` by default because result views may expose local paths.

## Repository Structure

```text
SemPIC/
├── run_generation_cache.py     # Build immutable teacher-cache artifacts
├── run_train.py                # Train SemPIC, KV Packet, or Joint adapters
├── run_eval.py                 # Evaluate cache reuse and recomputation methods
├── run_attention_analysis.py   # Collect and process attention diagnostics
├── run_eval_dashboard.py       # Browse and compare evaluation results
├── sempic/                     # Core library
├── generation_cache_config/    # Teacher-cache configs
├── train_config/               # Training configs
├── eval_config/                # Evaluation configs and stable results
├── attention_config/           # Attention-analysis configs
├── res/                        # Detailed configuration references
└── plot_scripts/               # Result visualization
```

## Documentation

- [Teacher generation-cache configuration](./res/generation_cache_config.md)
- [Training configuration](./res/train_config.md)
- [Evaluation methods and configuration](./res/eval_config.md)
- [Attention-analysis metrics and outputs](./res/attention_analysis.md)
- [Prompt sequence and reusable-block layout](./res/prompt_sequence.md)

## Supported Models

| Model family | Recommended template key |
| ------------ | ------------------------ |
| Llama 3.1    | `"tokenizer_chat"`       |
| Qwen3        | `"tokenizer_chat"`       |

Adding another RoPE model family requires model-specific KV re-rotation support in `sempic/cache_comb/recompute_kv/` and registration in `sempic/model/`.

## Citation

If you use SemPIC, please cite:

```bibtex
@misc{xie2026sempic,
  title={SemPIC: Learning Semantic Position-Independent KV Caches},
  author={Hui Xie and Peng Xiao and Yutong Deng and Shuoran Dou and Jian Yang and Jinyang Guo},
  year={2026},
  eprint={2607.28069},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2607.28069}
}
```

## Acknowledgements

The implementation and experimental framework build on [KV Packet](https://github.com/ChuangtaoChen-TUM/KVPacket) and follow its self-supervised distillation formulation. Please also cite the [KV Packet paper](https://arxiv.org/abs/2604.13226) when appropriate.
