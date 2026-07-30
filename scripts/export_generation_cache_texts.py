#!/usr/bin/env python3
"""Export generation-cache text with input and output token statistics."""

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from sempic.dataset import get_ret_eval_generator
from sempic.prompt import TokenizedPrompt, compile_prompt
from sempic.utils.generate import (
    generation_cache_key,
)
from sempic.utils.generation_cache_run import (
    GenerationCacheConfig,
    load_generation_cache_config,
)
from sempic.utils.generation_cache import load_generation_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export cached generated text as JSONL and report input/output token "
            "length statistics."
        )
    )
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--output-prefix",
        help="Output filename prefix; defaults to the cache filename stem.",
    )
    return parser.parse_args()


def load_prompts(
    config: GenerationCacheConfig,
    tokenizer: Any,
) -> list[tuple[str, TokenizedPrompt]]:
    prompts: list[tuple[str, TokenizedPrompt]] = []
    for data_config in config["data_configs"]:
        generator = get_ret_eval_generator(
            name=data_config["dataset_name"],
            num_samples=data_config["num_samples"],
            num_data_strs=data_config["num_data_strs"],
            num_shots=data_config["num_shots"],
            subset=data_config["subset"],
            split=data_config["split"],
            seed=data_config["seed"],
            data_kwargs=data_config["data_kwargs"],
            template=data_config["template"],
            template_kwargs=data_config["template_kwargs"],
            tokenizer=tokenizer,
        )
        prompts.extend(
            (
                generation_cache_key(sample["semantic"]),
                compile_prompt(tokenizer, sample["prompt"]),
            )
            for sample in generator
        )
    return prompts


def load_artifact_config(artifact_dir: Path) -> GenerationCacheConfig:
    sidecar_path = artifact_dir / "resolved_config.json"
    with sidecar_path.open(encoding="utf-8") as sidecar_file:
        return load_generation_cache_config(json.load(sidecar_file))


def max_new_tokens_from_config(config: GenerationCacheConfig) -> int:
    value = config["model"]["generation_kwargs"].get("max_new_tokens")
    if type(value) is not int or value < 1:
        raise ValueError(
            "resolved_config.json model.generation_kwargs.max_new_tokens "
            "must be a positive integer."
        )
    return value


def length_summary(lengths: list[int]) -> dict[str, float | int]:
    if not lengths:
        return {"count": 0}
    ordered = sorted(lengths)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (
            position - lower
        )

    return {
        "count": len(lengths),
        "min": min(lengths),
        "p25": round(percentile(0.25), 3),
        "median": round(statistics.median(lengths), 3),
        "mean": round(statistics.mean(lengths), 3),
        "p75": round(percentile(0.75), 3),
        "p95": round(percentile(0.95), 3),
        "max": max(lengths),
        "total": sum(lengths),
    }


def export_cache(
    artifact_dir: Path,
    config: GenerationCacheConfig,
    prompts: list[tuple[str, TokenizedPrompt]],
    max_new_tokens: int,
    tokenizer: Any,
    output_dir: Path,
    output_prefix: str,
) -> dict[str, Any]:
    cache = load_generation_cache(artifact_dir)
    cache_keys = cache.keys()
    prompt_by_key: dict[str, tuple[int, TokenizedPrompt]] = {}
    for sample_index, (key, prompt) in enumerate(prompts):
        prompt_by_key.setdefault(key, (sample_index, prompt))
    cache_key_set = set(cache_keys)
    prompt_key_set = set(prompt_by_key)
    if cache_key_set != prompt_key_set:
        missing = len(prompt_key_set - cache_key_set)
        unexpected = len(cache_key_set - prompt_key_set)
        raise ValueError(
            f"Cache/config prompt mismatch: {missing} missing and "
            f"{unexpected} unexpected cache keys."
        )

    dataset_names = sorted(
        {data_config["dataset_name"] for data_config in config["data_configs"]}
    )
    all_path = output_dir / f"{output_prefix}_texts.jsonl"
    max_path = output_dir / f"{output_prefix}_max_length_texts.jsonl"
    all_tmp = all_path.with_suffix(all_path.suffix + ".tmp")
    max_tmp = max_path.with_suffix(max_path.suffix + ".tmp")
    input_lengths: list[int] = []
    output_lengths: list[int] = []
    max_count = 0

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        with all_tmp.open("w", encoding="utf-8") as all_file, max_tmp.open(
            "w", encoding="utf-8"
        ) as max_file:
            for cache_index, cache_key in enumerate(cache_keys):
                sample_index, prompt = prompt_by_key[cache_key]
                metadata = cache.metadata(cache_key)
                assert metadata is not None
                input_token_length = int(prompt.input_ids.numel())
                input_text = tokenizer.decode(
                    prompt.input_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                input_lengths.append(input_token_length)
                for sequence_index, (output_token_length, text) in enumerate(zip(
                    metadata.sequence_lengths,
                    metadata.text,
                    strict=True,
                )):
                    output_lengths.append(output_token_length)
                    reached_max_length = output_token_length == max_new_tokens
                    record = {
                        "datasets": dataset_names,
                        "cache_index": cache_index,
                        "sample_index": sample_index,
                        "sequence_index": sequence_index,
                        "cache_key": cache_key,
                        "input_text": input_text,
                        "input_token_length": input_token_length,
                        "token_length": output_token_length,
                        "max_new_tokens": max_new_tokens,
                        "reached_max_length": reached_max_length,
                        "text": text,
                    }
                    line = json.dumps(record, ensure_ascii=False) + "\n"
                    all_file.write(line)
                    if reached_max_length:
                        max_file.write(line)
                        max_count += 1
            all_file.flush()
            os.fsync(all_file.fileno())
            max_file.flush()
            os.fsync(max_file.fileno())
        os.replace(all_tmp, all_path)
        os.replace(max_tmp, max_path)
    finally:
        all_tmp.unlink(missing_ok=True)
        max_tmp.unlink(missing_ok=True)

    return {
        "cache": str(artifact_dir),
        "all_texts": str(all_path),
        "max_length_texts": str(max_path),
        "cache_entries": len(cache),
        "mapping_mode": "semantic-sample-v2",
        "max_length_records": max_count,
        "input_tokens": length_summary(input_lengths),
        "output_tokens": length_summary(output_lengths),
    }


def main() -> None:
    args = parse_args()
    config = load_artifact_config(args.cache)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["tokenizer_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    prompts = load_prompts(config, tokenizer)
    max_new_tokens = max_new_tokens_from_config(config)
    output_dir = args.output_dir or args.cache.parent / f"{args.cache.name}_text_exports"
    output_prefix = args.output_prefix or args.cache.name
    summary = export_cache(
        args.cache,
        config,
        prompts,
        max_new_tokens,
        tokenizer,
        output_dir,
        output_prefix,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
