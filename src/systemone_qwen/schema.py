"""Request schema: the System One wire format (OpenRouter /api/v1/systemone, TypeSafe /v1/systemone)."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Instructions and criteria accept structure (TypeSafe "Advanced: structure").
Content = str | dict[str, Any] | list[Any]


class NoulCriteria(BaseModel):
    true: Content | None = None
    false: Content | None = None


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: Content
    criteria: NoulCriteria | None = None


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: Content
    # Option key -> description; null means the key speaks for itself. TypeSafe caps it at 255.
    criteria: dict[str, Content | None] = Field(min_length=2, max_length=255)


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: Content
    # Ordered levels; TypeSafe accepts 2 to 10.
    criteria: list[Content] = Field(min_length=2, max_length=10)


Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class DecisionRequest(BaseModel):
    # OpenRouter-only fields (provider, session_id, user, trace) are accepted and ignored.
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    state: Any
    questions: dict[str, Question] = Field(min_length=1, max_length=255)
