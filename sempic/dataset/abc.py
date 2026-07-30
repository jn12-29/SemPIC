from typing import Iterator, Protocol, TypeAlias, TypedDict

from ..prompt import (
    ContextBlock,
    Inline,
    PromptPart,
    PromptSequence,
    normalize_text_prompt,
)

JsonValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | None
    | list["JsonValue"]
    | dict[str, "JsonValue"]
)


class SemanticShot(TypedDict):
    documents: list[str]
    query: str
    answer: str
    task: dict[str, JsonValue]


class SemanticSample(TypedDict):
    documents: list[str]
    query: str
    shots: list[SemanticShot]
    task: dict[str, JsonValue]


class RetEvalEntry(TypedDict):
    """
    An evaluation entry for Retrieval-Augmented Generation (RAG) tasks.

    Attributes:
        prompt (PromptSequence[str]): The complete typed prompt.
        query (str): The query for the retrieval task.
        answer (str): The expected answer to the query.
        semantic (SemanticSample): Raw sample identity before rendering or tokenization.
    """
    prompt: PromptSequence[str]
    query: str
    answer: str
    semantic: SemanticSample


def build_prompt_sequence(
    preamble: str,
    documents: list[str],
    task_prompt: str,
) -> PromptSequence[str]:
    """Build the default prompt with explicit ordinary-text separators."""
    parts: list[PromptPart[str]] = [Inline(preamble)]
    for document_index, document in enumerate(documents):
        if document_index > 0:
            parts.append(Inline(" "))
        parts.append(ContextBlock(document))
    parts.extend((Inline(" "), Inline(task_prompt)))
    return normalize_text_prompt(PromptSequence(tuple(parts)))


class RetEvalGeneratorFunc(Protocol):
    def __call__(
        self,
        num_samples: int,
        num_data_strs: int,
        num_shots: int,
        subset: str,
        split: str,
        seed: int,
        **kwargs: dict
    ) -> Iterator[RetEvalEntry]:
        ...
