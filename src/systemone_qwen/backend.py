"""Client for one llama-server: OpenAI /v1/completions with logprobs, plus /tokenize.

Logprobs come back in two shapes: llama-server's `logprobs.content[0].top_logprobs` (a list
carrying token ids) and the legacy OpenAI/vLLM `logprobs.top_logprobs[0]` (a token -> logprob
map). Both are read. llama-server reports raw logprobs (before temperature), so the
requested sampling temperature does not change them.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import ModelConfig


class BackendError(RuntimeError):
    """The model server failed or answered something unusable (HTTP 502 to our caller)."""


@dataclass
class Alternative:
    token: str
    logprob: float
    id: int | None = None


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    async def add(self, payload: dict[str, Any]) -> None:
        usage = payload.get("usage") or {}
        async with self.lock:
            self.input_tokens += usage.get("prompt_tokens", 0) or 0
            self.output_tokens += usage.get("completion_tokens", 0) or 0


def first_token_alternatives(payload: dict[str, Any]) -> list[Alternative]:
    try:
        logprobs = payload["choices"][0]["logprobs"]
    except (KeyError, IndexError, TypeError) as error:
        raise BackendError(f"completion without logprobs: {str(payload)[:200]}") from error
    if not logprobs:
        raise BackendError("completion returned null logprobs; the backend must support `logprobs`")
    content = logprobs.get("content")
    alternatives: list[Alternative] = []
    if content:
        alternatives = [Alternative(e["token"], e["logprob"], e.get("id")) for e in content[0].get("top_logprobs") or []]
    elif logprobs.get("top_logprobs"):
        alternatives = [Alternative(token, logprob) for token, logprob in (logprobs["top_logprobs"][0] or {}).items()]
    if not alternatives:
        raise BackendError("completion returned no top logprobs")
    return alternatives


class Backend:
    def __init__(self, config: ModelConfig, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.config = config
        api_key = os.environ.get(config.api_key_env, "") if config.api_key_env else ""
        self.client = httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            timeout=httpx.Timeout(config.timeout_s),
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            transport=transport,
        )
        self.slots = asyncio.Semaphore(config.concurrency)
        self._token_cache: dict[str, list[tuple[int, str]]] = {}

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        async with self.slots:
            try:
                response = await self.client.post(path, json=body)
            except httpx.HTTPError as error:
                raise BackendError(f"{path}: {error}") from error
        if response.status_code != 200:
            raise BackendError(f"{path} returned {response.status_code}: {response.text[:300]}")
        return response.json()

    async def completion(self, prompt: str, usage: Usage, **body: Any) -> dict[str, Any]:
        request = {**self.config.extra_body, "prompt": prompt, **body}
        if self.config.served_model:
            request["model"] = self.config.served_model
        payload = await self._post("/v1/completions", request)
        await usage.add(payload)
        return payload

    async def first_token(self, prompt: str, usage: Usage, top: int) -> list[Alternative]:
        payload = await self.completion(prompt, usage, max_tokens=1, temperature=0, logprobs=top)
        return first_token_alternatives(payload)

    async def think(self, prompt: str, usage: Usage) -> str:
        """Reasoning text, with Qwen's recommended thinking sampler, cut at </think> or the budget."""
        payload = await self.completion(
            prompt, usage, max_tokens=self.config.think_tokens, temperature=0.6, top_p=0.95, seed=0,
            stop=["</think>"],
        )
        return payload["choices"][0].get("text") or ""

    async def tokenize(self, text: str) -> list[tuple[int, str]]:
        """llama-server /tokenize -> [(id, piece)]; only reachable with direct server access."""
        if text not in self._token_cache:
            payload = await self._post("/tokenize", {"content": text, "add_special": False, "with_pieces": True})
            tokens = payload.get("tokens") or []
            self._token_cache[text] = [(t["id"], t.get("piece", "")) if isinstance(t, dict) else (t, "")
                                       for t in tokens]
        return self._token_cache[text]

    async def healthy(self) -> bool:
        try:
            response = await self.client.get("/health", timeout=3)
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def close(self) -> None:
        await self.client.aclose()
