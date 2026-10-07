"""Questions -> answer distributions -> System One answers.

Each question is one prompt whose options are lettered A, B, C...; the answer distribution is
the softmax of each code's logprob as the model's first answer token. Codes beyond Z need the
server's tokenizer (llama-server /tokenize) and one extra request per shared token prefix.
"""

from __future__ import annotations

import asyncio
import math
import uuid
from typing import Any

from .backend import Alternative, Backend, BackendError, Usage
from .config import ModelConfig, Settings
from .prompt import chatml, codes, options, render, user_message
from .schema import DecisionRequest, NoulQuestion, Question, ScoreQuestion


class UnsupportedRequest(ValueError):
    """Valid request this backend cannot answer faithfully (HTTP 422)."""


def softmax(scores: list[float]) -> list[float]:
    top = max(scores)
    weights = [math.exp(s - top) for s in scores]
    total = sum(weights)
    return [w / total for w in weights]


def calibrate(probabilities: list[float], temperature: float) -> list[float]:
    """p ∝ p^(1/T): T > 1 softens, the argmax (hence accuracy) never changes."""
    if temperature == 1.0:
        return probabilities
    return softmax([math.log(max(p, 1e-12)) / temperature for p in probabilities])


def confidence(probabilities: list[float]) -> float:
    """TypeSafe's confidence: the top probability rescaled so a uniform distribution is 0.

    Equal to (n·p_max − 1)/(n − 1), i.e. p_max minus the mean of the others; it matches Jev's
    published answers and Contrastive-LM's reimplementation.
    """
    n = len(probabilities)
    return max(0.0, min(1.0, (n * max(probabilities) - 1) / (n - 1)))


def _by_token(alternatives: list[Alternative]) -> tuple[dict[str, float], float]:
    seen: dict[str, float] = {}
    for alt in alternatives:
        seen.setdefault(alt.token, alt.logprob)
    return seen, min(alt.logprob for alt in alternatives)


def _by_id(alternatives: list[Alternative]) -> tuple[dict[int, float], float]:
    seen = {alt.id: alt.logprob for alt in alternatives if alt.id is not None}
    if not seen:
        raise UnsupportedRequest("more than 26 options needs token ids in logprobs (direct llama-server access)")
    return seen, min(alt.logprob for alt in alternatives)


async def code_logprobs(backend: Backend, prompt: str, letters: list[str], usage: Usage) -> list[float]:
    """Log-probability of each code as the answer. An option absent from the returned top-N gets
    the lowest returned logprob, an upper bound on its true value."""
    top = max(backend.config.top_logprobs, min(1000, 4 * len(letters)))
    first = await backend.first_token(prompt, usage, top)
    if all(len(code) == 1 for code in letters):
        seen, floor = _by_token(first)
        return [seen.get(code, floor) for code in letters]

    try:
        pieces = [await backend.tokenize(code) for code in letters]
    except BackendError as error:
        raise UnsupportedRequest(f"more than 26 options needs llama-server /tokenize: {error}") from error
    cache: dict[tuple[int, ...], tuple[dict[int, float], float]] = {(): _by_id(first)}

    async def distribution(prefix: list[tuple[int, str]]) -> tuple[dict[int, float], float]:
        key = tuple(token_id for token_id, _ in prefix)
        if key not in cache:
            text = prompt + "".join(piece for _, piece in prefix)
            cache[key] = _by_id(await backend.first_token(text, usage, top))
        return cache[key]

    scores = []
    for tokens in pieces:
        total = 0.0
        for index, (token_id, _) in enumerate(tokens):
            seen, floor = await distribution(tokens[:index])
            total += seen.get(token_id, floor)
        scores.append(total)
    return scores


async def option_distribution(
    backend: Backend, config: ModelConfig, state: Any, question: Question, labels: list[str], usage: Usage
) -> list[float]:
    letters = codes(len(labels))
    user = user_message(state, question.instructions, list(zip(letters, labels, strict=True)))
    thinking = ""
    if config.think_tokens:
        thinking = await backend.think(chatml(config.system_prompt, user, thinking=None), usage)
    prompt = chatml(config.system_prompt, user, thinking)
    return softmax(await code_logprobs(backend, prompt, letters, usage))


async def answer(backend: Backend, config: ModelConfig, state: Any, question: Question, usage: Usage) -> dict:
    keys, labels = options(question, config)
    probabilities = await option_distribution(backend, config, state, question, labels, usage)
    if isinstance(question, NoulQuestion) and config.debias:
        # Small models favour a position regardless of content; averaging both orders cancels it.
        backward = await option_distribution(backend, config, state, question, labels[::-1], usage)
        probabilities = [(a + b) / 2 for a, b in zip(probabilities, backward[::-1], strict=True)]
    probabilities = calibrate(probabilities, config.calibration_temperature)

    if isinstance(question, NoulQuestion):
        return {"type": "noul", "noul": probabilities[0]}
    distribution = dict(zip(keys, probabilities, strict=True))
    if isinstance(question, ScoreQuestion):
        return {
            "type": "score",
            # Probability-weighted level, as TypeSafe documents it: it can land between levels.
            "score": sum(level * p for level, p in enumerate(probabilities)),
            "legend": {str(level): render(c) for level, c in enumerate(question.criteria)},
            "probabilities": distribution,
            "confidence": confidence(probabilities),
        }
    return {
        "type": "choice",
        "choice": keys[max(range(len(keys)), key=probabilities.__getitem__)],
        "probabilities": distribution,
        "confidence": confidence(probabilities),
    }


async def decide(settings: Settings, backends: dict[str, Backend], request: DecisionRequest) -> dict:
    name, config = settings.resolve(request.model)
    backend = backends[name]
    usage = Usage()
    ids = list(request.questions)
    results = await asyncio.gather(*(answer(backend, config, request.state, request.questions[i], usage) for i in ids))
    return {
        "id": f"gen-dec-{uuid.uuid4().hex}",
        "model": name,
        "provider": settings.provider,
        "answers": dict(zip(ids, results, strict=True)),
        "usage": {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens, "cost": 0},
    }
