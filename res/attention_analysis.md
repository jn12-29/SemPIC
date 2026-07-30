# Attention Analysis

The attention pipeline has three independent stages:

1. `run_attention_analysis.py` performs real model forwards and streams each layer basis through the configured reducers.
2. `run_process_attention.py` derives plot-ready metrics from saved statistics without loading a model.
3. `plot_scripts/draw_attention_analysis.py` re-renders one processed variant.

## Collection contract

An analysis config contains query passes, not figures. The paper config runs `terminal_query`, `gold_answer`, and `shifted_prediction` (the causal rows that predict the gold-answer tokens).

For each `sample × method × query pass`, the collector performs one real query forward. Terminal collection uses the same shared interleaved layout and compact Inline execution as evaluation, with an observable eager attention backend in place of the normal FlexAttention backend. All compact Inline rows still participate in the real layer-to-layer hidden-state computation; query selection limits only the rows streamed to reducers. There is no second terminal-only attention recomputation.

Every layer exposes the selected rows' ephemeral scaled masked logits and post-softmax probabilities together with the complete physical Value states, keep mask, GQA query-to-KV-head mapping, and read-only physical chunk projection. The source basis tensors are released after the synchronous callbacks and are never saved. To preserve exact Full-relative absolute deviation under method-major forwards, `attention_profile` keeps CPU copies of the current sample's Full chunk probabilities in their original dtype until all candidates finish; normalization and aggregation use FP32.

The default reducers are:

- `attention_profile`: Full raw and chunk-conditional attention profiles, plus each candidate's aligned absolute deviation;
- `pic_retrieval`: per-query-head retrieval-vector NRMSE, cosine distance, and reusable-scope attention-mass error.

The optional `raw_attention_profile` reducer stores each method's own raw
attention density over canonical reusable-block tokens. For every layer and
token, it averages post-softmax probabilities equally over the selected query
heads and query rows, writes only a CPU FP32 `[layer, token]` tensor, and does
not retain a cross-method Full reference. It does not apply chunk-conditional
renormalization or include PacketWrapper filler positions.

Retrieval is computed as attention probability times Value independently for every query head. Full, Vanilla PIC, and SemPIC use the canonical PIC token scope. KVPacket and SemPIC + KVPacket use wrapper head + PIC + wrapper tail. Query heads are never averaged before retrieval comparison, and `o_proj` is not applied.

Every model/dataset group requires `full_recompute`. Candidate method keys are `vanilla_pic` (eval method `no_recompute`), `kvpacket`, `sempic`, and optional `sempic_kvpacket`. A single invocation may contain multiple models and datasets; datasets remain separate estimates in the combined model figures.

## Run

Check idle GPUs immediately before model loading, then bind one physical GPU:

```bash
nvidia-smi

CUDA_VISIBLE_DEVICES=<physical_gpu_id> uv run python run_attention_analysis.py \
  eval_config/qwen_3_4b/biography/full_recompute.json \
  eval_config/qwen_3_4b/biography/no_recompute.json \
  --analysis-config attention_config/paper_attention.json \
  --processing-config attention_config/processing_default.json \
  --output-dir attention_results \
  --run-name paper \
  --cpu-threads 4
```

After training, create the artifact-backed eval configs described in [eval_config.md](./eval_config.md) and add their paths to the same command. Add the corresponding method configs for every other dataset/model included in the run. Every model/dataset group requires `full_recompute`; candidate method keys are `vanilla_pic` (eval method `no_recompute`), `kvpacket`, `sempic`, and optionally `sempic_kvpacket`. Use `--max-samples N` only for a controlled subset; omit it for the configured full datasets. Compression and KV quantization are rejected because canonical token alignment requires one physical PIC position per source token.

The attention-sink preset is `attention_config/paper_attention_sink.json`. It
collects only `shifted_prediction` for matched Full Recompute, No Recompute, and
SemPIC configs. Include those three configs for each model/dataset group after
creating the artifact-backed SemPIC config. The existing reducers keep the
standard processing reports viable; `raw_attention_profile` supplies the
per-method source values for the separate sink export.

Resume an interrupted run using only its saved config:

```bash
CUDA_VISIBLE_DEVICES=<physical_gpu_id> uv run python run_attention_analysis.py \
  --resume-run attention_results/<run-directory>
```

The run freezes mutable artifact aliases to concrete paths. It does not recursively scan model or adapter trees. A complete reduced checkpoint is atomically committed after all methods and reducers finish one sample/query pass; an interrupted current sample/pass is rerun.

## Outputs

```text
attention_results/<timestamp>_paper_attention[_name]/
├── config.json
├── statistics/<model>/<dataset>/<query_pass>.pt
├── .work/<model>/<dataset>/<query_pass>/sample_XXXXXX.pt
└── analysis/<timestamp>-<number-or-suffix>/
    ├── processing_config.json
    ├── metrics.pt
    └── figures/<model>/
        ├── attention_maps.pdf
        ├── attention_error_structure.pdf
        ├── retrieval_nrmse.pdf
        ├── retrieval_cosine_distance.pdf
        └── retrieval_mass_error.pdf
```

Each partition stores all configured reducers and methods for one model, dataset, and query pass. It contains reduced statistics tensors only—never QK logits, Values, per-query retrieval vectors, or full attention blocks. Per-sample checkpoints use the same reduced content and enable resume.

The partition identity freezes the evaluation top-level `eval_seed` separately
from `dataset_config.seed`; consumers must not assume those two seeds are equal.
Partitions without `eval_seed` have missing behavior provenance, and consumers
must not infer a value.

For sink-density exports, region density is computed from raw profiles by
averaging eligible tokens within each chunk and layer, weighting chunks equally
within each sample, weighting layers equally, and finally weighting samples
equally. The default sink regions use each token's normalized-width overlap
with the leading 10% and middle 80%, so boundary tokens contribute only their
eligible width. Normalized-position curves use the same equal-chunk,
equal-layer, and equal-sample hierarchy after token-width binning.

`metrics.pt` contains flat, self-describing records. Each model receives five PDF-only, multi-page reports. Datasets occupy separate rows and are never averaged together. Attention reports use one page per query pass and attention view; retrieval reports use one page per query pass. Missing optional methods are shown as `N/A`. Plotting fails if a required report family, matching layer/global record, or coordinate alignment is incomplete.

The tensor-valued records remain compact and directly reusable by the plotting frontend. Processing does not expand every layer/head/position value into a large CSV. Small paper-table summaries should be exported separately from the relevant global or regional records when needed.

Heatmaps use one robust color scale per dataset, shared by comparable methods in that dataset. The display maximum is the 99.5th percentile; an extended colorbar reports both that threshold and the true maximum whenever values are clipped. This keeps attention sinks or isolated head/layer spikes from hiding the remaining structure without silently discarding the extrema. Extreme-dynamic-range error matrices and retrieval NRMSE views use a labeled `symlog` scale; cosine distance keeps its natural `[0, 2]` range. Other views retain adaptive nonnegative linear axes.

`attention_maps.pdf` places Full Recompute attention beside every candidate's Full-relative deviation. `attention_error_structure.pdf` condenses Prefix/Interior/Suffix error for every configured edge ratio into a layer matrix plus a separate mean strip. Each retrieval report combines per-head heatmaps, global `mean ± SEM`, and overlaid layer-wise method curves on the same dataset row.

## Reprocess and replot

```bash
uv run python run_process_attention.py \
  --run-dir attention_results/<run-directory> \
  --processing-config attention_config/processing_default.json \
  --suffix middle-focus \
  --cpu-threads 4

uv run python plot_scripts/draw_attention_analysis.py \
  attention_results/<run-directory>/analysis/<variant-directory>
```

Every processing call creates a new timestamped variant. Automatic variants receive `-0`, `-1`, and so on; a manual suffix is also supported. Existing variants are preserved.

`position_mode` is `absolute`, `normalized`, or `auto`; auto uses absolute positions only when all chunk lengths match. `edge_ratios` controls separate prefix, interior, and suffix summaries at several widths. Prefix and suffix are never folded into a normalized boundary-distance coordinate. Processing weights chunks equally within a sample and samples equally for mean/SEM.
