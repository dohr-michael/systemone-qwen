"""Model registry: which llama-server answers which `model` name, and how to ask it.

Configuration comes from a YAML file (SYSTEMONE_CONFIG) or, for a single model, from
environment variables. See config.example.yaml for every field.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    # llama-server root URL (direct access enables /tokenize, hence more than 26 options).
    base_url: str
    # `model` sent to the backend: llama-server ignores it, LiteLLM routes on it.
    served_model: str | None = None
    # Name of the environment variable holding the backend API key, if any.
    api_key_env: str | None = None
    # How noul options are worded: "Yes"/"No", the criteria, or "Yes: <criterion>".
    noul_labels: Literal["plain", "criteria", "prefixed"] = "prefixed"
    # Average noul probabilities over both option orders; only helps sub-1B models.
    debias: bool = False
    # p ∝ p^(1/T) on each answer distribution; >1 softens over-confident models.
    calibration_temperature: float = Field(default=1.0, gt=0)
    # >0: let the model reason up to that many tokens before reading the answer.
    think_tokens: int = Field(default=0, ge=0)
    # First-token alternatives requested; options missing from them get the lowest returned logprob.
    top_logprobs: int = Field(default=100, ge=2)
    # Requests run in parallel per request against this backend (match llama-server --parallel).
    concurrency: int = Field(default=1, ge=1)
    timeout_s: float = Field(default=120.0, gt=0)
    # Extra fields merged into every completion request (llama-server: reuse the KV prefix).
    extra_body: dict = Field(default_factory=lambda: {"cache_prompt": True})
    system_prompt: str = "Choose one option. Answer only with its code."


class Settings(BaseModel):
    models: dict[str, ModelConfig]
    # Extra names accepted in `model`, e.g. {"jev-latest": "qwen-decision"}.
    aliases: dict[str, str] = Field(default_factory=dict)
    # Used when a request names no model, or one that is not configured, if allow_unknown_model.
    default_model: str | None = None
    allow_unknown_model: bool = False
    # Name reported in the response's `provider` field.
    provider: str = "systemone-qwen"

    @model_validator(mode="after")
    def check_references(self) -> "Settings":
        if not self.models:
            raise ValueError("at least one model must be configured")
        for alias, target in self.aliases.items():
            if target not in self.models:
                raise ValueError(f"alias {alias!r} points to unknown model {target!r}")
        if self.default_model is None:
            self.default_model = next(iter(self.models))
        elif self.default_model not in self.models:
            raise ValueError(f"default_model {self.default_model!r} is not configured")
        return self

    def resolve(self, name: str | None) -> tuple[str, ModelConfig]:
        """-> (canonical name, config); KeyError when the name is unknown and not allowed."""
        if name:
            name = self.aliases.get(name, name)
            if name in self.models:
                return name, self.models[name]
            if not self.allow_unknown_model:
                raise KeyError(name)
        default = self.default_model or next(iter(self.models))
        return default, self.models[default]


def load_settings() -> Settings:
    path = os.environ.get("SYSTEMONE_CONFIG")
    if path:
        return Settings.model_validate(yaml.safe_load(Path(path).read_text()) or {})
    base_url = os.environ.get("SYSTEMONE_BACKEND_URL")
    if not base_url:
        raise RuntimeError("Set SYSTEMONE_CONFIG (YAML file) or SYSTEMONE_BACKEND_URL (single llama-server).")
    name = os.environ.get("SYSTEMONE_MODEL", "qwen-decision")
    fields = {
        "base_url": base_url,
        "served_model": os.environ.get("SYSTEMONE_SERVED_MODEL"),
        "api_key_env": os.environ.get("SYSTEMONE_API_KEY_ENV"),
    }
    return Settings(models={name: ModelConfig(**{k: v for k, v in fields.items() if v})})
