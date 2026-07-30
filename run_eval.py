import os
import argparse
import glob
import logging
import statistics
from pathlib import Path
from contextlib import nullcontext
from typing import TypedDict, Iterator, Callable
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers import GenerationConfig
from sempic.cache.compress import ScorerPress
from sempic.packet_wrapper import PacketWrapper
from sempic.cache_comb import build_cache_comb_executor
from sempic.cache_comb.runtime import TTFTTimer, generate_from_prefill
from sempic.dataset import ANSWER_POSTPROCESS_DICT
from sempic.dataset.abc import RetEvalEntry
from sempic.evaluation import (
    LORA_EVAL_ADAPTER_NAME,
    EvalConfig,
    EvalResourceCache,
    QuantizationConfig,
    build_compressor,
    build_eval_generator,
    build_generation_config,
    create_eval_resource_cache,
    load_eval_config,
    load_eval_resources,
    load_lora_adapter_for_eval,
    prepare_prompt_kvs,
    prepared_source_parts as _prepared_source_parts,
)
from sempic.utils.metric import calculate_metrics
from sempic.utils.generate import get_answers
from sempic.utils.metric import f1_states
from sempic.utils.config import gather_config_files, load_config_file
from sempic.utils.lora import lora_adapters_disabled
from sempic.utils.runtime import (
    DebugRecorder,
    RuntimeCLIOverrides,
    RuntimeContext,
    add_runtime_cli_args,
    apply_runtime_overrides,
    debug_recording_scope,
    runtime_overrides_from_args,
)
from sempic.utils.run_storage import (
    allocate_run_dir,
    atomic_write_json,
    compose_run_suffix,
    validate_run_suffix,
)
from sempic.model import SupportedModel
from sempic.prompt import compile_prompt

LOGGER = logging.getLogger("sempic.run_eval")


class _NoEvalConfigsFound(Exception):
    pass


class EvalResult(TypedDict):
    precision: float
    recall: float
    f1: float
    ttft: float
    ttft_mean: float
    ttft_p50: float
    ttft_p90: float
    ttft_p99: float
    ttft_min: float
    ttft_max: float
    ttft_std: float
    ttft_count: int
    flops: float
    num_orig_tokens: int
    num_wrapped_tokens: int


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _summarize_ttft(values: list[float]) -> dict[str, float | int]:
    mean = statistics.fmean(values) if values else 0.0
    return {
        "ttft": mean,
        "ttft_mean": mean,
        "ttft_p50": _percentile(values, 0.50),
        "ttft_p90": _percentile(values, 0.90),
        "ttft_p99": _percentile(values, 0.99),
        "ttft_min": min(values, default=0.0),
        "ttft_max": max(values, default=0.0),
        "ttft_std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "ttft_count": len(values),
    }


def _safe_system_segment(value: str) -> str:
    return compose_run_suffix(value) or "run"


def eval_config_scope(
    eval_config_file: str | Path,
    eval_config_root: str | Path = "eval_config",
) -> Path:
    config_path = Path(os.path.abspath(eval_config_file))
    config_root = Path(os.path.abspath(eval_config_root))
    try:
        relative_parent = config_path.relative_to(config_root).parent
    except ValueError:
        return Path(
            "_external",
            _safe_system_segment(config_path.parent.name),
            _safe_system_segment(config_path.stem),
        )
    if relative_parent == Path("."):
        return Path("root")
    return Path(*(_safe_system_segment(part) for part in relative_parent.parts))


def stable_eval_result_path(eval_config_file: str | Path) -> Path:
    config_path = Path(eval_config_file)
    return config_path.parent / "eval_results" / f"{config_path.stem}_result.json"


def run_eval(
    model: SupportedModel,
    tokenizer: PreTrainedTokenizer|PreTrainedTokenizerFast,
    eval_generator: Iterator[RetEvalEntry],
    cache_comb_method: str,
    cache_comb_kwargs: dict,
    packet_wrapper: PacketWrapper|None = None,
    compressor: ScorerPress|None = None,
    keep_filler_tokens: bool = False,
    quantization_config: QuantizationConfig|None = None,
    generation_config: GenerationConfig|None = None,
    answer_postprocess_func: Callable[[str, str], tuple[str, str]]|None = None,
    lora_adapter_name: str | None = None,
    debug_recorder: DebugRecorder|None = None,
) -> EvalResult:
    cache_comb_func = build_cache_comb_executor(cache_comb_method, model)
    total_tp = 0
    total_fp = 0
    total_fn = 0
    ttft_values: list[float] = []
    total_flops = 0.0
    num_orig_tokens: int = 0
    num_wrapped_tokens: int = 0

    num_eval = 0
    for eval_entry in eval_generator:
        query = eval_entry["query"]
        gt_answer = eval_entry["answer"]
        prompt = compile_prompt(tokenizer, eval_entry["prompt"]).to(model.device)
        sample_debug = (
            debug_recorder.sample_scope("sample", num_eval)
            if debug_recorder is not None
            else None
        )
        if sample_debug is not None and not sample_debug.enabled:
            sample_debug = None
        if sample_debug is not None:
            sample_debug.record_json(
                "formatted_segments",
                {
                    "cache_comb_method": cache_comb_method,
                    "query": query,
                    "answer": gt_answer,
                    "prompt_parts": [
                        {"kind": span.kind, "start": span.start, "end": span.end}
                        for span in prompt.parts
                    ],
                    "lengths": {
                        "query_chars": len(query),
                        "answer_chars": len(gt_answer),
                        "prompt_tokens": int(prompt.input_ids.numel()),
                    },
                },
            )

        prepared = prepare_prompt_kvs(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            method_name=cache_comb_method,
            packet_wrapper=packet_wrapper,
            compressor=compressor,
            keep_filler_tokens=keep_filler_tokens,
            quantization_config=quantization_config,
            lora_adapter_name=lora_adapter_name,
            debug_recorder=sample_debug,
        )
        prepared_kvs = prepared.prepared_kvs
        num_orig_tokens += prepared.num_orig_tokens
        num_wrapped_tokens += prepared.num_wrapped_tokens

        effective_answer_postprocess_func = answer_postprocess_func
        if sample_debug is not None:
            def debug_answer_postprocess(
                pred_answer: str,
                answer: str,
                *,
                _base_func: Callable[[str, str], tuple[str, str]]|None = answer_postprocess_func,
                _recorder: DebugRecorder = sample_debug,
            ) -> tuple[str, str]:
                post_pred = pred_answer
                post_answer = answer
                if _base_func is not None:
                    post_pred, post_answer = _base_func(pred_answer, answer)
                _recorder.record_json(
                    "prediction",
                    {
                        "prediction_raw": pred_answer,
                        "answer_raw": answer,
                        "prediction": post_pred,
                        "answer": post_answer,
                        "postprocess_applied": _base_func is not None,
                    },
                )
                return post_pred, post_answer
            effective_answer_postprocess_func = debug_answer_postprocess

        lora_generation_context = (
            lora_adapters_disabled(model)
            if lora_adapter_name is not None
            else nullcontext()
        )
        with lora_generation_context:
            with debug_recording_scope(sample_debug):
                warmup = getattr(cache_comb_func, "warmup", None)
                if callable(warmup):
                    warmup(
                        model=model,
                        tokenizer=tokenizer,
                        generation_config=generation_config,
                        prompt=prompt,
                        prepared_kvs=prepared_kvs,
                        answer=gt_answer,
                        answer_postprocess_func=effective_answer_postprocess_func,
                        kwargs=cache_comb_kwargs,
                    )
                ttft_timer = TTFTTimer(model.device)
                ttft_timer.start()
                prefill = cache_comb_func(
                    model=model,
                    tokenizer=tokenizer,
                    generation_config=generation_config,
                    prompt=prompt,
                    prepared_kvs=prepared_kvs,
                    answer=gt_answer,
                    answer_postprocess_func=effective_answer_postprocess_func,
                    kwargs=cache_comb_kwargs,
                )
                generation, ttft = generate_from_prefill(
                    model=model,
                    tokenizer=tokenizer,
                    generation_config=generation_config,
                    result=prefill,
                    ttft_timer=ttft_timer,
                )

        pred_answer = get_answers(
            generation, prefill.generation_input_ids, tokenizer
        )[0]
        answer = gt_answer
        if effective_answer_postprocess_func is not None:
            pred_answer, answer = effective_answer_postprocess_func(pred_answer, answer)
        tp, fp, fn = f1_states(
            gold_tokens=answer.split(), pred_tokens=pred_answer.split()
        )
        flops = prefill.flops

        total_tp += tp
        total_fp += fp
        total_fn += fn
        ttft_values.append(ttft)
        total_flops += flops
        num_eval += 1
        if sample_debug is not None:
            sample_debug.record_json(
                "result",
                {
                    "ttft": ttft,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "flops": flops,
                },
            )
    
    precision, recall, f1 = calculate_metrics(total_tp, total_fp, total_fn)
    avg_flops = total_flops / num_eval if num_eval > 0 else 0.0
    ttft_summary = _summarize_ttft(ttft_values)

    f_result = EvalResult(
        precision=precision,
        recall=recall,
        f1=f1,
        **ttft_summary,
        flops=avg_flops,
        num_orig_tokens=num_orig_tokens,
        num_wrapped_tokens=num_wrapped_tokens
    )

    return f_result


def run_one_config(
    eval_config_file: str,
    eval_cache: EvalResourceCache,
    eval_results: dict[str, dict],
    overwrite: bool = False,
    runtime_overrides: RuntimeCLIOverrides|None = None,
    cli_args: dict[str, object]|None = None,
    config_index: int|None = None,
    config_count: int|None = None,
    discovered_configs: list[str]|None = None,
    discovery_warnings: list[str]|None = None,
    invocation_logger: logging.Logger | None = None,
    eval_output_root: str | Path = "eval_outputs",
    eval_config_root: str | Path = "eval_config",
    run_suffix_override: str | None = None,
):
    del config_index, config_count, discovered_configs, discovery_warnings
    result_path = stable_eval_result_path(eval_config_file)
    result_file = result_path.name
    event_logger = invocation_logger or LOGGER

    if not overwrite and result_path.exists():
        event_logger.info("Skipping existing evaluation for config: %s", eval_config_file)
        return

    eval_config_json = load_config_file(
        eval_config_file,
        default_config_file="_default.json"
    )
    eval_config_json = apply_runtime_overrides(eval_config_json, runtime_overrides)
    eval_config = load_eval_config(eval_config_json)
    configured_suffix = eval_config_json.get("run_suffix")
    effective_suffix = (
        run_suffix_override
        if run_suffix_override is not None
        else configured_suffix
    )
    validate_run_suffix(effective_suffix)

    config_stem = _safe_system_segment(Path(eval_config_file).stem)
    method = _safe_system_segment(eval_config["cache_comb"]["method"])
    run_suffix = compose_run_suffix(
        config_stem,
        user_suffix=effective_suffix,
    )
    run_root = (
        Path(eval_output_root)
        / eval_config_scope(eval_config_file, eval_config_root)
        / method
    )
    run_dir = allocate_run_dir(run_root, run_suffix)

    eval_snapshot = dict(eval_config)
    if effective_suffix is not None:
        eval_snapshot["run_suffix"] = effective_suffix

    with RuntimeContext(
        entrypoint="run_eval",
        run_dir=run_dir,
        config_file=eval_config_file,
        resolved_config=eval_snapshot,
        config_snapshot_name="eval_config.json",
        cli_args={} if cli_args is None else cli_args,
    ) as runtime_context:
        runtime_context.logger.info("Running evaluation for config: %s", eval_config_file)

        lora_path = eval_config["lora"]["path"]
        resources = load_eval_resources(eval_config, eval_cache)
        model = resources.model
        tokenizer = resources.tokenizer
        packet_wrapper = resources.packet_wrapper
        eval_generator = build_eval_generator(eval_config["dataset"], tokenizer)

        answer_postprocess_func = ANSWER_POSTPROCESS_DICT.get(
            eval_config["dataset"]["dataset_name"], None
        )
        generation_config = build_generation_config(eval_config["model"])

        cache_comb_method = eval_config["cache_comb"]["method"]
        comb_kwargs = eval_config["cache_comb"].get("kwargs", {})

        compressor, keep_filler_tokens = build_compressor(eval_config["compress"])

        # Run evaluation
        assert isinstance(model, SupportedModel), "Model type not supported."
        result = run_eval(
            model=model,
            tokenizer=tokenizer,
            eval_generator=eval_generator,
            cache_comb_method=cache_comb_method,
            cache_comb_kwargs=comb_kwargs,
            packet_wrapper=packet_wrapper,
            compressor=compressor,
            keep_filler_tokens=keep_filler_tokens,
            quantization_config=eval_config["quantization"],
            generation_config=generation_config,
            answer_postprocess_func=answer_postprocess_func,
            lora_adapter_name=getattr(resources, "lora_adapter_name", None),
            debug_recorder=runtime_context.debug_recorder,
        )

        payload = {
            "config": eval_snapshot,
            "result": result,
        }
        eval_results[eval_config_file] = payload
        canonical_result_path = runtime_context.run_dir / result_file
        atomic_write_json(canonical_result_path, payload)
        atomic_write_json(result_path, payload)
        runtime_context.logger.info("Evaluation for config %s completed.", eval_config_file)



def main(
    argv: list[str] | None = None,
    *,
    eval_output_root: str | Path = "eval_outputs",
    eval_config_root: str | Path = "eval_config",
) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_files_or_paths",
        type=str,
        nargs="+",
        help="Path to the evaluation configuration file or directory (glob pattern)."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results."
    )
    parser.add_argument(
        "--run-suffix",
        type=str,
        default=None,
        help="Append a recognizable suffix to invocation and config run names.",
    )
    add_runtime_cli_args(parser)
    args = parser.parse_args(argv)

    config_files_or_paths: list[str] = args.config_files_or_paths
    overwrite: bool = args.overwrite
    runtime_overrides = runtime_overrides_from_args(args)
    cli_args = vars(args).copy()
    assert isinstance(config_files_or_paths, list)
    validate_run_suffix(args.run_suffix)

    invocation_dir = allocate_run_dir(
        Path(eval_output_root) / "_invocations",
        args.run_suffix,
    )

    try:
        with RuntimeContext(
            entrypoint="run_eval",
            run_dir=invocation_dir,
            config_file=None,
            resolved_config=None,
            config_snapshot_name=None,
            cli_args=cli_args,
        ) as invocation_context:
            all_config_files: set[str] = set()
            discovery_warnings: list[str] = []

            for pattern in config_files_or_paths:
                matched_paths = glob.glob(pattern, recursive=False)

                for path in matched_paths:
                    try:
                        configs = gather_config_files(
                            path,
                            pattern=r"\.json$",
                            skip_pattern=r"_default\.json"
                        )
                        for c in configs:
                            all_config_files.add(c)
                    except ValueError as e:
                        discovery_warnings.append(f"{e} Skipping path: {path}")

            sorted_config_files = sorted(all_config_files)
            invocation_context.logger.info(
                "Found %d configuration files:", len(sorted_config_files)
            )
            for warning in discovery_warnings:
                invocation_context.logger.warning("%s", warning)
            for discovered_config in sorted_config_files:
                invocation_context.logger.info("  %s", discovered_config)

            if not sorted_config_files:
                invocation_context.logger.error(
                    "No configuration files found. Please check the provided paths and patterns."
                )
                raise _NoEvalConfigsFound

            invocation_context.logger.info("Starting evaluation.")
            eval_results: dict[str, dict] = {}
            eval_cache = create_eval_resource_cache()

            for eval_config_file in sorted_config_files:
                run_one_config(
                    eval_config_file,
                    eval_cache,
                    eval_results,
                    overwrite=overwrite,
                    runtime_overrides=runtime_overrides,
                    cli_args=cli_args,
                    invocation_logger=invocation_context.logger,
                    eval_output_root=eval_output_root,
                    eval_config_root=eval_config_root,
                    run_suffix_override=args.run_suffix,
                )
    except _NoEvalConfigsFound:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
