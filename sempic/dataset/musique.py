import random
import warnings
import re
from typing import Iterator
from datasets import load_dataset, Dataset
from .abc import RetEvalEntry, SemanticShot, build_prompt_sequence

MUSIQUE_DATASET: dict[str, Dataset|None] = {}


def musique_ret_eval_generator(
    num_samples: int,
    num_data_strs: int,
    num_shots: int,
    subset: str = "default",
    split: str = "train",
    seed: int = 42,
    **kwargs
) -> Iterator[RetEvalEntry]:
    cache_dataset = kwargs.pop("cache_dataset", True)
    add_inst = kwargs.pop("add_inst", True)
    add_cot = kwargs.pop("add_cot", True)

    if kwargs:
        warnings.warn(f"Unused kwargs in musique_ret_eval_generator: {kwargs}")

    if cache_dataset:
        cached_ds = MUSIQUE_DATASET.get(subset)
    else:
        cached_ds = None

    if cached_ds is not None:
        ds = cached_ds
    else:
        ds = load_dataset("dgslibisey/MuSiQue", split=split)
        assert isinstance(ds, Dataset)
    
    ds_len = len(ds)
    all_indices = list(range(ds_len))

    rng = random.Random(seed)
    rng.shuffle(all_indices)

    few_shot_indices = all_indices[-num_shots:] if num_shots > 0 else []
    
    if ds_len - num_shots < num_samples:
        raise ValueError(
            f"Not enough samples in the dataset after accounting for few-shots: "
            f"requested {num_samples} samples but only {ds_len - num_shots} available."
        )
    
    selected_indices = all_indices[: ds_len - num_shots]

    few_shot_strs: list[str] = []
    semantic_shots: list[SemanticShot] = []

    if num_shots > 0 and add_cot:
        warnings.warn("Do not recommend using few-shot with CoT for MuSiQue dataset.")

    for idx in few_shot_indices:
        sample = ds[idx]
        paragraphs = sample["paragraphs"]
        docs = "\n".join(para["paragraph_text"] for para in paragraphs)
        question = sample["question"]
        answer = sample["answer"]

        if add_inst:
            question = "Answer the following question based on the provided context.\n" + question
        else:
            question = "Question: " + question

        few_shot_str = f"Context: {docs}\n{question}\nAnswer: {answer}\n"
        few_shot_strs.append(few_shot_str)
        semantic_shots.append(SemanticShot(
            documents=[para["paragraph_text"] for para in paragraphs],
            query=sample["question"],
            answer=answer,
            task={},
        ))
    

    if few_shot_strs:
        few_shot_str = "\n".join(few_shot_strs) + "\n"
    else:
        few_shot_str = ""

    
    for i in range(num_samples):
        sample_idx = selected_indices[i]
        sample = ds[sample_idx]
        paragraphs = sample["paragraphs"]
        data_strs = [
            para["paragraph_text"] for para in paragraphs
        ]
        question = sample["question"]
        answer_str = sample["answer"]

        if add_inst:
            question = "Answer the following question based on the provided context.\n" + question + "\n"
        if add_cot:
            question_str = "You should get the final answer by thinking step by step."
            question_str += "\nThe answer is a word or short phrase. Your response should end with: 'Answer: <your final answer>'.\n"
            question = question + question_str

        yield RetEvalEntry(
            prompt=build_prompt_sequence(few_shot_str, data_strs, question),
            query=sample["question"],
            answer=answer_str,
            semantic={
                "documents": data_strs,
                "query": sample["question"],
                "shots": list(semantic_shots),
                "task": {},
            },
        )


def musique_answer_postprocess(pred_answer: str, gold_answer: str) -> tuple[str, str]:
    parts = re.split(r'(?i)Answer\s*:', pred_answer)
    if len(parts) > 1:
        # We take the last part to get the content after the separator
        pred_answer = parts[-1].lower().strip().rstrip('.')
    else:
        pred_answer = ""

    gold_answer = gold_answer.lower().strip().rstrip('.')
    return pred_answer, gold_answer
