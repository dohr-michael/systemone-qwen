"""API behaviour against a fake llama-server (httpx.MockTransport): no model, no network."""

from __future__ import annotations

import json
import math
import re

import httpx
import pytest
from fastapi.testclient import TestClient

from systemone_qwen.app import create_app
from systemone_qwen.backend import Backend
from systemone_qwen.config import ModelConfig, Settings
from systemone_qwen.prompt import chatml, codes, user_message


class FakeLlamaServer:
    """Answers /v1/completions with chosen first-token logprobs and records every request.

    `logprobs` maps an option code (A, B, ...) to its logprob; a code missing from it is left
    out of the returned top-N. `legacy=True` returns the OpenAI/vLLM `top_logprobs` map shape.
    """

    def __init__(self, logprobs=None, legacy=False, status=200, thought="It depends on the receipt."):
        self.logprobs = logprobs or {"A": -0.1, "B": -2.5}
        self.legacy, self.status, self.thought = legacy, status, thought
        self.requests: list[tuple[str, dict]] = []
        self.ids = {}  # token piece -> id, for /tokenize and id-carrying logprobs

    def token_id(self, piece: str) -> int:
        return self.ids.setdefault(piece, 1000 + len(self.ids))

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        self.requests.append((request.url.path, body))
        if self.status != 200:
            return httpx.Response(self.status, text="boom")
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/tokenize":
            # Single letters are one token; two-letter codes split into two letters.
            return httpx.Response(200, json={"tokens": [{"id": self.token_id(c), "piece": c} for c in body["content"]]})
        if body.get("max_tokens", 1) > 1:  # thinking pass
            return httpx.Response(200, json={"choices": [{"text": self.thought}],
                                             "usage": {"prompt_tokens": 50, "completion_tokens": 7}})
        table = self.logprobs(body["prompt"]) if callable(self.logprobs) else self.logprobs
        if self.legacy:
            logprobs = {"top_logprobs": [dict(table)], "content": None}
        else:
            logprobs = {"content": [{"token": "A", "logprob": 0.0, "top_logprobs": [
                {"id": self.token_id(t), "token": t, "logprob": lp} for t, lp in table.items()]}]}
        return httpx.Response(200, json={"choices": [{"text": "A", "logprobs": logprobs}],
                                         "usage": {"prompt_tokens": 40, "completion_tokens": 1}})


def client_for(fake: FakeLlamaServer, **config) -> TestClient:
    model = ModelConfig(base_url="http://llama", **config)
    settings = Settings(models={"qwen-decision": model}, aliases={"jev-latest": "qwen-decision"})
    backends = {"qwen-decision": Backend(model, transport=httpx.MockTransport(fake))}
    return TestClient(create_app(settings, backends))


def ask(client: TestClient, questions: dict, path="/api/v1/systemone", **extra) -> httpx.Response:
    return client.post(path, json={"model": "qwen-decision", "state": "I was charged twice.", "questions": questions,
                                   **extra})


def softmax2(a: float, b: float) -> float:
    return math.exp(a) / (math.exp(a) + math.exp(b))


NOUL = {"type": "noul", "instructions": "Does the customer ask for a refund?",
        "criteria": {"true": "Asks for money back", "false": "Does not"}}


def test_noul_returns_probability_of_the_yes_option():
    fake = FakeLlamaServer()
    with client_for(fake) as client:
        response = ask(client, {"refund": NOUL})
    assert response.status_code == 200
    body = response.json()
    assert body["answers"]["refund"] == {"type": "noul", "noul": pytest.approx(softmax2(-0.1, -2.5))}
    assert body["model"] == "qwen-decision" and body["provider"] == "systemone-qwen"
    assert body["id"].startswith("gen-dec-")
    assert body["usage"] == {"input_tokens": 40, "output_tokens": 1, "cost": 0}
    prompt = fake.requests[0][1]["prompt"]
    assert "A. Yes: Asks for money back\nB. No: Does not" in prompt
    assert fake.requests[0][1]["cache_prompt"] is True and fake.requests[0][1]["max_tokens"] == 1


def test_choice_shows_keys_and_reports_confidence():
    fake = FakeLlamaServer({"A": -0.2, "B": -1.9, "C": -4.0})
    question = {"type": "choice", "instructions": "Which team?",
                "criteria": {"billing": "Payments, refunds", "technical": "Bugs", "sales": None}}
    with client_for(fake) as client:
        answer = ask(client, {"team": question}).json()["answers"]["team"]
    assert answer["choice"] == "billing"
    assert list(answer["probabilities"]) == ["billing", "technical", "sales"]
    assert sum(answer["probabilities"].values()) == pytest.approx(1)
    top = answer["probabilities"]["billing"]
    assert answer["confidence"] == pytest.approx((3 * top - 1) / 2)
    assert "A. billing: Payments, refunds\nB. technical: Bugs\nC. sales" in fake.requests[0][1]["prompt"]


def test_score_is_the_expected_level_with_legend():
    fake = FakeLlamaServer({"A": -5.0, "B": -1.0, "C": -0.5})
    question = {"type": "score", "instructions": "How urgent?", "criteria": ["not urgent", "soon", "critical"]}
    with client_for(fake) as client:
        answer = ask(client, {"urgency": question}).json()["answers"]["urgency"]
    p = answer["probabilities"]
    assert answer["score"] == pytest.approx(p["1"] + 2 * p["2"])
    assert answer["legend"] == {"0": "not urgent", "1": "soon", "2": "critical"}


def test_option_missing_from_top_n_gets_the_lowest_returned_logprob():
    fake = FakeLlamaServer({"A": -0.1, "B": -3.0})
    question = {"type": "choice", "instructions": "?", "criteria": {"x": None, "y": None, "z": None}}
    with client_for(fake) as client:
        p = ask(client, {"q": question}).json()["answers"]["q"]["probabilities"]
    assert p["y"] == pytest.approx(p["z"])


def test_legacy_logprob_shape_is_read():
    fake = FakeLlamaServer(legacy=True)
    with client_for(fake) as client:
        answer = ask(client, {"refund": NOUL}).json()["answers"]["refund"]
    assert answer["noul"] == pytest.approx(softmax2(-0.1, -2.5))


def test_calibration_softens_without_changing_the_choice():
    fake = FakeLlamaServer({"A": -0.05, "B": -3.0})
    question = {"type": "choice", "instructions": "?", "criteria": {"x": None, "y": None}}
    with client_for(fake) as raw_client:
        raw = ask(raw_client, {"q": question}).json()["answers"]["q"]
    with client_for(FakeLlamaServer({"A": -0.05, "B": -3.0}), calibration_temperature=3.0) as cal_client:
        calibrated = ask(cal_client, {"q": question}).json()["answers"]["q"]
    assert calibrated["choice"] == raw["choice"] == "x"
    assert 0.5 < calibrated["probabilities"]["x"] < raw["probabilities"]["x"]


def test_debias_averages_both_option_orders():
    # The model always prefers the first position: a pure position bias.
    fake = FakeLlamaServer({"A": -0.1, "B": -2.5})
    with client_for(fake, debias=True) as client:
        answer = ask(client, {"refund": NOUL}).json()["answers"]["refund"]
    assert answer["noul"] == pytest.approx(0.5)
    assert "A. No: Does not\nB. Yes: Asks for money back" in fake.requests[1][1]["prompt"]


def test_thinking_generates_reasoning_then_reads_the_answer_after_it():
    fake = FakeLlamaServer()
    with client_for(fake, think_tokens=256) as client:
        body = ask(client, {"refund": NOUL}).json()
    think, read = fake.requests[0][1], fake.requests[1][1]
    assert think["max_tokens"] == 256 and think["stop"] == ["</think>"] and think["prompt"].endswith("<think>\n")
    assert read["prompt"].endswith("<think>\nIt depends on the receipt.\n</think>\n\n")
    assert body["usage"]["output_tokens"] == 7 + 1


def test_more_than_26_options_uses_tokenize_and_prefix_requests():
    letters = codes(28)  # A..Z, AA, AB
    fake = FakeLlamaServer()

    def table(prompt: str) -> dict:
        # After the prompt: "A" likely; after an extra "A": second letter "B" likely.
        return {"B": -0.2, "A": -2.0} if prompt.endswith("\n\nA") else {"A": -0.3, "C": -1.5}

    fake.logprobs = table
    question = {"type": "choice", "instructions": "?", "criteria": {f"o{i}": None for i in range(28)}}
    with client_for(fake) as client:
        answer = ask(client, {"q": question}).json()["answers"]["q"]
    assert any(path == "/tokenize" for path, _ in fake.requests)
    # "AB" = P(A) · P(B | A) and "AA" = P(A) · P(A | A).
    p = answer["probabilities"]
    assert p["o27"] > p["o26"]
    assert letters[26:] == ["AA", "AB"]


def test_more_than_26_options_without_tokenize_is_a_clear_422():
    fake = FakeLlamaServer()
    real = fake.__call__

    def no_tokenize(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404) if request.url.path == "/tokenize" else real(request)

    model = ModelConfig(base_url="http://litellm")
    settings = Settings(models={"m": model})
    app = create_app(settings, {"m": Backend(model, transport=httpx.MockTransport(no_tokenize))})
    question = {"type": "choice", "instructions": "?", "criteria": {f"o{i}": None for i in range(30)}}
    with TestClient(app) as client:
        response = client.post("/v1/systemone", json={"state": "s", "questions": {"q": question}})
    assert response.status_code == 422 and "/tokenize" in response.json()["error"]["message"]


def test_aliases_paths_and_models_listing():
    with client_for(FakeLlamaServer()) as client:
        for path in ("/api/v1/systemone", "/v1/systemone", "/api/alpha/decisions"):
            assert ask(client, {"r": NOUL}, path=path, model="jev-latest").json()["model"] == "qwen-decision"
        ids = {m["id"] for m in client.get("/v1/models").json()["data"]}
        assert ids == {"qwen-decision", "jev-latest"}
        assert client.get("/ready").json() == {"backends": {"qwen-decision": True}}


def test_errors_map_to_http_statuses():
    with client_for(FakeLlamaServer()) as client:
        assert ask(client, {"r": NOUL}, model="nope").status_code == 404
        bad = {"type": "score", "instructions": "?", "criteria": ["only one"]}
        assert ask(client, {"r": bad}).status_code == 422
        ignored = ask(client, {"r": NOUL}, provider={"order": ["x"]}, session_id="s", user="u")
        assert ignored.status_code == 200
    with client_for(FakeLlamaServer(status=500)) as client:
        assert ask(client, {"r": NOUL}).status_code == 502


def test_prompt_is_byte_identical_to_the_benchmarked_lab_version():
    # Exactly what decision_bench.QwenServer sent during the JevBench runs in docs/benchmarks.md.
    expected = ('<|im_start|>system\nChoose one option. Answer only with its code.\n<|im_end|>\n'
                '<|im_start|>user\nState: {"msg":"hi"}\n\nQuestion: Greeting?\n\nOptions:\nA. Yes\nB. No'
                '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
    user = user_message({"msg": "hi"}, "Greeting?", [("A", "Yes"), ("B", "No")])
    assert chatml("Choose one option. Answer only with its code.", user) == expected
    assert re.search(r"<think>\n$", chatml("s", "u", thinking=None))
