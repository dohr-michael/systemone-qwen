"""Prompt construction, kept byte-identical to the benchmarked lab version (see docs/benchmarks.md).

The model sees the state, the question and lettered options, and answers with one code; the
answer is read from the first token's logprobs, never generated. Thinking is closed unless a
reasoning budget is configured.
"""

from __future__ import annotations

import json
from itertools import product
from typing import Any

from .config import ModelConfig
from .schema import ChoiceQuestion, NoulQuestion, Question

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def render(value: Any) -> str:
    """Strings verbatim, structure as compact JSON."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def codes(count: int) -> list[str]:
    """A..Z, then AA, AB... for larger option sets."""
    out: list[str] = []
    width = 1
    while len(out) < count:
        out.extend("".join(chars) for chars in product(ALPHABET, repeat=width))
        width += 1
    return out[:count]


def options(question: Question, config: ModelConfig) -> tuple[list[str], list[str]]:
    """-> (answer keys in order, option labels shown to the model)."""
    if isinstance(question, NoulQuestion):
        criteria = question.criteria
        true, false = (criteria.true, criteria.false) if criteria else (None, None)
        if config.noul_labels == "plain" or true is None or false is None:
            labels = ["Yes", "No"]
        elif config.noul_labels == "prefixed":
            labels = [f"Yes: {render(true)}", f"No: {render(false)}"]
        else:
            labels = [render(true), render(false)]
        return ["true", "false"], labels
    if isinstance(question, ChoiceQuestion):
        keys = list(question.criteria)
        # Key and description, as Jev sees them: keys often carry meaning ("billing").
        labels = [key if question.criteria[key] in (None, "") else f"{key}: {render(question.criteria[key])}"
                  for key in keys]
        return keys, labels
    return [str(level) for level in range(len(question.criteria))], [render(c) for c in question.criteria]


def user_message(state: Any, instructions: Any, lettered: list[tuple[str, str]]) -> str:
    rendered = "\n".join(f"{code}. {label}" for code, label in lettered)
    state_json = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    return f"State: {state_json}\n\nQuestion: {render(instructions)}\n\nOptions:\n{rendered}"


def chatml(system: str, user: str, thinking: str | None = "") -> str:
    """Qwen ChatML. thinking="" closes an empty think block, a string closes it after that
    reasoning, and None leaves it open so the model can generate its reasoning first."""
    prompt = (
        f"<|im_start|>system\n{system}\n<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )
    if thinking is not None:
        prompt += f"{thinking.strip()}\n</think>\n\n" if thinking.strip() else "\n</think>\n\n"
    return prompt
