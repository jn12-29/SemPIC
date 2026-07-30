from typing import Any, Protocol

from ..prompt import ContextBlock, Inline, PromptPart, PromptSequence, normalize_text_prompt
from .abc import RetEvalEntry


TOKENIZER_CHAT_SENTINEL = "<|sempic_segment_split|>"


def _tokenizer_chat_boundary_sentinel(index: int) -> str:
    return f"{TOKENIZER_CHAT_SENTINEL}[{index}]"


class TemplateFunc(Protocol):
    def __call__(
        self,
        eval_entry: RetEvalEntry,
        tokenizer: Any|None = None,
        **kwargs,
    ) -> RetEvalEntry:
        """Format an entry's typed prompt for an instruction model."""
        ...


def default_template(
    eval_entry: RetEvalEntry,
    **kwargs,
) -> RetEvalEntry:
    """Return the typed prompt unchanged."""
    return eval_entry


def llama_chat_template(
    eval_entry: RetEvalEntry,
    system_prompt: str = "",
    **kwargs,
) -> RetEvalEntry:
    begin_str = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n"
    if system_prompt:
        begin_str += f"{system_prompt}"

    begin_str += "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    end_str = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    return _wrap_prompt(eval_entry, begin_str, end_str)


def qwen_3_chat_template(
    eval_entry: RetEvalEntry,
    system_prompt: str = "",
    **kwargs,
) -> RetEvalEntry:
    """Apply the Qwen 3 (and 2.5) chat wrapper."""
    begin_str = ""

    if system_prompt:
        begin_str += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"

    begin_str += "<|im_start|>user\n"

    end_str = "<|im_end|>\n<|im_start|>assistant\n"
    return _wrap_prompt(eval_entry, begin_str, end_str)


def _wrap_prompt(
    eval_entry: RetEvalEntry,
    prefix: str,
    suffix: str,
) -> RetEvalEntry:
    prompt = eval_entry["prompt"]
    terminal = prompt.parts[-1]
    assert isinstance(terminal, Inline)
    wrapped_prompt = normalize_text_prompt(PromptSequence((
        Inline(prefix),
        *prompt.parts[:-1],
        Inline(terminal.content + suffix),
    )))
    return RetEvalEntry(
        prompt=wrapped_prompt,
        query=eval_entry["query"],
        answer=eval_entry["answer"],
        semantic=eval_entry["semantic"],
    )


def tokenizer_chat_template(
    eval_entry: RetEvalEntry,
    tokenizer: Any|None = None,
    system_prompt: str = "",
    enable_thinking: bool = False,
    **kwargs,
) -> RetEvalEntry:
    if kwargs:
        unknown_keys = ", ".join(sorted(kwargs.keys()))
        raise ValueError(f"Unknown tokenizer_chat template kwargs: {unknown_keys}")

    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise ValueError("tokenizer_chat template requires a tokenizer with apply_chat_template().")

    prompt = eval_entry["prompt"]
    prompt_parts: tuple[PromptPart[str], ...]
    if isinstance(prompt.parts[0], Inline):
        prompt_parts = prompt.parts
    else:
        prompt_parts = (Inline(""), *prompt.parts)

    if TOKENIZER_CHAT_SENTINEL in system_prompt:
        raise ValueError("tokenizer_chat sentinel collision in system_prompt.")
    for part_index, part in enumerate(prompt_parts):
        if TOKENIZER_CHAT_SENTINEL in part.content:
            raise ValueError(
                f"tokenizer_chat sentinel collision in prompt.parts[{part_index}]."
            )

    def render_user_content(user_content: str) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": user_content,
        })
        rendered_chat = apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        if not isinstance(rendered_chat, str):
            raise ValueError("tokenizer_chat template expected apply_chat_template(..., tokenize=False) to return str.")
        return rendered_chat

    boundary_sentinels = [
        _tokenizer_chat_boundary_sentinel(idx)
        for idx in range(len(prompt_parts) - 1)
    ]
    user_content_parts = [prompt_parts[0].content]
    for sentinel, part in zip(boundary_sentinels, prompt_parts[1:]):
        user_content_parts.extend([sentinel, part.content])
    user_content = "".join(user_content_parts)
    rendered = render_user_content(user_content)

    sentinel_positions = []
    for sentinel in boundary_sentinels:
        sentinel_count = rendered.count(sentinel)
        if sentinel_count != 1:
            raise ValueError(
                "tokenizer_chat sentinel count mismatch after applying chat template: "
                f"{sentinel} expected 1, got {sentinel_count}."
            )
        sentinel_positions.append(rendered.index(sentinel))
    if sentinel_positions != sorted(sentinel_positions):
        raise ValueError("tokenizer_chat rendered sentinels are out of order.")

    rendered_segments: list[str] = []
    segment_start = 0
    for sentinel, sentinel_pos in zip(boundary_sentinels, sentinel_positions):
        rendered_segments.append(rendered[segment_start:sentinel_pos])
        segment_start = sentinel_pos + len(sentinel)
    rendered_segments.append(rendered[segment_start:])

    rendered_parts: list[PromptPart[str]] = []
    for original_part, rendered_segment in zip(
        prompt_parts,
        rendered_segments,
        strict=True,
    ):
        if isinstance(original_part, Inline):
            rendered_parts.append(Inline(rendered_segment))
        else:
            if rendered_segment != original_part.content:
                raise ValueError(
                    "tokenizer_chat template modified ContextBlock content at "
                    f"prompt.parts[{len(rendered_parts)}]."
                )
            rendered_parts.append(original_part)

    return RetEvalEntry(
        prompt=normalize_text_prompt(PromptSequence(tuple(rendered_parts))),
        query=eval_entry["query"],
        answer=eval_entry["answer"],
        semantic=eval_entry["semantic"],
    )


TEMPLATE_FUNC_DICT: dict[str, TemplateFunc] = {
    "default": default_template,
    "llama_chat": llama_chat_template,
    "qwen_3_chat": qwen_3_chat_template,
    "tokenizer_chat": tokenizer_chat_template,
}
