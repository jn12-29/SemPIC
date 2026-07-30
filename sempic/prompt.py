from dataclasses import dataclass
from typing import Any, Generic, Iterable, Literal, TypeAlias, TypeVar

import torch


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Inline(Generic[T]):
    content: T


@dataclass(frozen=True, slots=True)
class ContextBlock(Generic[T]):
    content: T


PromptPart: TypeAlias = Inline[T] | ContextBlock[T]


@dataclass(frozen=True, slots=True)
class PromptSequence(Generic[T]):
    parts: tuple[PromptPart[T], ...]

    def __post_init__(self) -> None:
        if not self.parts or not isinstance(self.parts[-1], Inline):
            raise ValueError("PromptSequence must end with an Inline part.")
        if isinstance(self.parts[-1].content, str) and not self.parts[-1].content:
            raise ValueError("PromptSequence must end with a non-empty Inline part.")


@dataclass(frozen=True, slots=True)
class TokenSpan:
    kind: Literal["inline", "context"]
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class TokenizedPrompt:
    input_ids: torch.Tensor
    parts: tuple[TokenSpan, ...]

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 1:
            raise ValueError("TokenizedPrompt.input_ids must be one-dimensional.")
        if not self.parts:
            raise ValueError("TokenizedPrompt must contain at least one part.")

        cursor = 0
        for span in self.parts:
            if span.kind not in ("inline", "context"):
                raise ValueError(f"Unsupported token span kind: {span.kind!r}.")
            if span.start != cursor or span.end < span.start:
                raise ValueError("Token spans must form an ordered, gap-free partition.")
            if span.kind == "context" and span.end == span.start:
                raise ValueError("ContextBlock spans must be non-empty.")
            cursor = span.end
        if cursor != self.input_ids.numel():
            raise ValueError("Token spans must cover the canonical input IDs exactly.")

        final_span = self.parts[-1]
        if final_span.kind != "inline" or final_span.end == final_span.start:
            raise ValueError("TokenizedPrompt must end with a non-empty Inline span.")

    def to(self, device: torch.device | str) -> "TokenizedPrompt":
        return TokenizedPrompt(input_ids=self.input_ids.to(device), parts=self.parts)


def normalize_text_prompt(prompt: PromptSequence[str]) -> PromptSequence[str]:
    parts: list[PromptPart[str]] = []
    for part in prompt.parts:
        if isinstance(part, Inline):
            if not part.content:
                continue
            if parts and isinstance(parts[-1], Inline):
                previous = parts[-1]
                parts[-1] = Inline(previous.content + part.content)
            else:
                parts.append(part)
        else:
            parts.append(part)
    return PromptSequence(tuple(parts))


def compile_prompt(tokenizer: Any, prompt: PromptSequence[str]) -> TokenizedPrompt:
    normalized = normalize_text_prompt(prompt)
    encoded = tokenizer(
        [part.content for part in normalized.parts],
        add_special_tokens=False,
        padding=False,
    )
    encoded_parts: Iterable[Any] = encoded["input_ids"]

    flat_ids: list[int] = []
    spans: list[TokenSpan] = []
    for part, part_ids in zip(normalized.parts, encoded_parts, strict=True):
        if isinstance(part_ids, torch.Tensor):
            ids = part_ids.tolist()
        else:
            ids = list(part_ids)
        start = len(flat_ids)
        flat_ids.extend(ids)
        spans.append(TokenSpan(
            kind="inline" if isinstance(part, Inline) else "context",
            start=start,
            end=len(flat_ids),
        ))

    return TokenizedPrompt(
        input_ids=torch.tensor(flat_ids, dtype=torch.long),
        parts=tuple(spans),
    )
