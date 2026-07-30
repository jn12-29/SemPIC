from typing import Iterator
import warnings
import random
import re
from datasets import load_dataset, Dataset
from .abc import RetEvalEntry, SemanticShot, build_prompt_sequence


def hotpot_qa_ret_eval_generator(
    num_samples: int,
    num_data_strs: int,
    num_shots: int,
    subset: str = "fullwiki",
    split: str = "validation",
    seed: int = 42,
    **kwargs
) -> Iterator[RetEvalEntry]:
    """
    Generates evaluation entries for the HotpotQA dataset. Each document is the concatenation
    of sentences from the context provided in the dataset.

    Args:
        num_samples (int): Number of samples to generate.
        num_data_strs (int): Number of data strings to include in each entry (not used here).
        num_shots (int): Number of few-shot examples to include in the preamble.
        subset (str): Subset of the HotpotQA dataset to use ("fullwiki" or "distractor").
        split (str): Split of the dataset to use ("train", "validation", or "test").
        seed (int): Random seed for shuffling the dataset.
        **kwargs: Additional keyword arguments.
        - add_instruction (bool): Whether to add instructions to the question prompt. Default is True.
        - add_cot (bool): Whether to add chain-of-thought prompting to the question prompt. Default is True.
        - difficulty (list[str]): Difficulty levels to include. Defaults to ["hard"].
    
    Yields:
        RetEvalEntry: An evaluation entry containing a typed prompt, query, and answer.
    
    Note:
        - The hotpot QA dataset should be used by an instruction-tuned model as it requires multi-hop reasoning.
        - Few-shot prompting is not recommended for this dataset as the multi-hop content is missing in the context.
        - num_data_strs is not used in this generator since each document has its own sentences. The number of documents
            is determined by the number of sentences in the context.
    """
    add_inst = kwargs.pop("add_inst", True)
    add_cot = kwargs.pop("add_cot", True)
    difficulty = kwargs.pop("difficulty", ["hard"])

    if (
        not isinstance(difficulty, list)
        or not difficulty
        or not all(isinstance(level, str) for level in difficulty)
    ):
        raise ValueError("difficulty must be a non-empty list of strings.")
    unknown_difficulties = set(difficulty) - {"easy", "medium", "hard"}
    if unknown_difficulties:
        raise ValueError(f"Unknown difficulty levels: {sorted(unknown_difficulties)}")
    if kwargs:
        warnings.warn(f"Unused kwargs in hotpot_qa_eval_generator: {kwargs}")

    ds_split = load_dataset("hotpotqa/hotpot_qa", subset, split=split)
    assert isinstance(ds_split, Dataset)

    # ds_split = ds[split]
    ds_len = len(ds_split)

    all_indices = list(range(ds_len))
    rng = random.Random(seed)
    rng.shuffle(all_indices)
    all_indices = [
        index for index in all_indices
        if ds_split[index]["level"] in difficulty
    ]

    def format_data_str(index: int) -> list[str]:
        item = ds_split[index]
        context = item['context']
        sentences = context['sentences']
        flattened_sentences = [
            "".join(sent) for sent in sentences
        ]
        return flattened_sentences

    if num_shots > 0:
        warnings.warn("few_shot_str is not recommended for HotpotQA")
        few_shot_indices = all_indices[:num_shots]
        all_indices = all_indices[num_shots:]
        few_shot_strs = []
        semantic_shots: list[SemanticShot] = []
        for idx in few_shot_indices:
            item = ds_split[idx]
            question = item['question']
            answer = item['answer']
            context_strs = format_data_str(idx)
            few_shot_str = f"Context: {' '.join(context_strs)}\nQuestion: {question}\nAnswer: {answer}\n"
            few_shot_strs.append(few_shot_str)
            semantic_shots.append(SemanticShot(
                documents=context_strs,
                query=question,
                answer=answer,
                task={},
            ))
        few_shot_str = "\n".join(few_shot_strs)
    else:
        few_shot_str = ""
        semantic_shots = []

    if num_samples > len(all_indices):
        warnings.warn(f"num_samples ({num_samples}) is greater than dataset size ({len(all_indices)}). Reducing num_samples to dataset size.")
        num_samples = len(all_indices)

    for idx in all_indices[:num_samples]:
        item = ds_split[idx]
        question_str = item['question']
        if add_inst:
            question_str = f"Answer the following question based on the provided context.\n{question_str}"
        if add_cot:
            question_str += "You should get the final answer by thinking step by step.\n"
        if add_inst:
            question_str += "Your response should end with: 'Short Answer: <your final answer>'.\n"

        answer_str = item['answer']
        data_strs = format_data_str(idx)
        yield RetEvalEntry(
            prompt=build_prompt_sequence(few_shot_str, data_strs, question_str),
            query=item['question'],
            answer=answer_str,
            semantic={
                "documents": data_strs,
                "query": item['question'],
                "shots": list(semantic_shots),
                "task": {},
            },
        )


def hotpot_qa_answer_postprocess(pred_answer: str, gold_answer: str) -> tuple[str, str]:
    parts = re.split(r'(?i)Answer\s*:', pred_answer)
    if len(parts) > 1:
        # We take the last part to get the content after the separator
        pred_answer = parts[-1].lower().strip().rstrip('.')
    else:
        pred_answer = ""

    gold_answer = gold_answer.lower().strip().rstrip('.')
    return pred_answer, gold_answer
