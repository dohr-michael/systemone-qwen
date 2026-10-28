"""Records parity.json from the Python service (v0.1.0, commit 44f1083), the version benchmarked
in docs/benchmarks.md. The Go port must send byte-identical prompts and return the same answers.

Kept for provenance: it needs that commit checked out (`git worktree add /tmp/py 44f1083`), then
`cd /tmp/py && uv run python <this file> > <repo>/internal/systemone/testdata/parity.json`.

The fake llama-server is deterministic and mirrored in parity_test.go: each letter's logprob is
derived from sha256(prompt, letter), some letters are left out of the top-N (they get the
floor), and filler tokens pad the list to the requested size.
"""

import hashlib
import json

import httpx
from fastapi.testclient import TestClient

from systemone_qwen.app import create_app
from systemone_qwen.backend import Backend
from systemone_qwen.config import ModelConfig, Settings

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
THOUGHT = "  It depends on the receipt.\n  "


def digest(prompt: str, token: str) -> bytes:
    return hashlib.sha256((prompt + "\x00" + token).encode()).digest()


class Fake:
    def __init__(self, status=200):
        self.status, self.calls = status, []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        if request.url.path != "/health":
            self.calls.append([request.url.path, body])
        if self.status != 200:
            return httpx.Response(self.status, text="boom")
        if body.get("max_tokens", 1) > 1:
            return httpx.Response(200, json={"choices": [{"text": THOUGHT}],
                                             "usage": {"prompt_tokens": 50, "completion_tokens": 7}})
        prompt = body["prompt"]
        entries = []
        for letter in LETTERS:
            h = digest(prompt, letter)
            if h[4] % 5:
                entries.append((letter, -8 * int.from_bytes(h[:4], "big") / 2**32))
        filler = 0
        while len(entries) < body["logprobs"]:
            entries.append((f"~{filler}", -20 - filler / 100))
            filler += 1
        top = [{"id": 1000 + i, "token": t, "logprob": lp} for i, (t, lp) in enumerate(entries)]
        return httpx.Response(200, json={
            "choices": [{"text": entries[0][0], "logprobs": {"content": [
                {"id": 1000, "token": entries[0][0], "logprob": entries[0][1], "top_logprobs": top}]}}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 1}})


NOUL = {"type": "noul", "instructions": "Does the customer ask for a refund?",
        "criteria": {"true": "Asks for money back", "false": "Does not"}}
RICH_STATE = (r'{"z":1,"a":{"y":[1e2,1.50,-0,1e-5,1E16,1e15,12345678901234567890,true,null,'
              r'"tab\there \"q\" \\ \u0001 \u007f é é 🎉   <b>&amp; \/"],"x":{},"w":[]},'
              r'"dup":1,"m":[{"role":"user","content":"hi"},{"role":"assistant","content":"yo"}],"dup":2}')


def request(questions, state='"I was charged twice."', model="qwen-decision"):
    return f'{{"model":{json.dumps(model)},"state":{state},"questions":{json.dumps(questions, ensure_ascii=False)}}}'


CASES = [
    ("noul_prefixed", {}, request({"refund": NOUL})),
    ("noul_plain_rich_state", {"noul_labels": "plain"},
     request({"refund": {"type": "noul", "instructions": "Remboursement demandé ?"}}, RICH_STATE, "jev-latest")),
    ("choice_and_score_calibrated", {"calibration_temperature": 1.5}, request({
        "department": {"type": "choice", "instructions": {"task": "route", "lang": "fr"},
                       "criteria": {"technical": "Bugs", "billing": None, "other": "",
                                    "sales": {"examples": ["quote", "upgrade"]}}},
        "anger": {"type": "score", "instructions": "How angry?",
                  "criteria": ["Calm", {"level": "angry", "signs": ["caps"]}, ["very", "angry"]]}})),
    ("debias_and_thinking", {"debias": True, "think_tokens": 512}, request({
        "refund": NOUL,
        "team": {"type": "choice", "instructions": "Which team?", "criteria": {"a": "A team", "b": "B team"}}})),
    ("criteria_labels_custom_backend",
     {"noul_labels": "criteria", "served_model": "qwen-decision", "system_prompt": "Pick one.",
      "extra_body": {"cache_prompt": False, "n_keep": 0}},
     request({"refund": {"type": "noul", "instructions": "Refund?",
                         "criteria": {"true": {"says": "refund"}, "false": "No refund"}}})),
    ("question_order", {}, request({k: {"type": "noul", "instructions": f"Is it {k}?"}
                                    for k in ("zeta", "alpha", "mid", "beta", "omega")})),
    ("choice_26_options", {}, request({"pick": {"type": "choice", "instructions": "Pick",
                                                "criteria": {f"o{i}": f"option {i}" for i in range(26)}}})),
    # Errors: only the status is compared.
    ("unknown_model", {}, request({"r": NOUL}, model="nope")),
    ("score_one_level", {}, request({"r": {"type": "score", "instructions": "?", "criteria": ["one"]}})),
    ("box_question", {}, request({"r": {"type": "box", "instructions": "Where?"}})),
    ("image_in_state", {}, request({"r": NOUL}, '[{"type":"image_url","image_url":{"url":"data:,"}}]')),
    ("missing_state", {}, '{"questions":{"r":{"type":"noul","instructions":"?"}}}'),
    ("no_questions", {}, request({})),
    ("choice_one_option", {}, request({"r": {"type": "choice", "instructions": "?", "criteria": {"a": None}}})),
    ("numeric_instructions", {}, request({"r": {"type": "noul", "instructions": 5}})),
    ("unknown_type", {}, request({"r": {"type": "rank", "instructions": "?"}})),
    ("invalid_json", {}, '{"state": "s", "questions": '),
    ("backend_down", {"backend_status": 500}, request({"r": NOUL})),
]


def run(name, overrides, raw):
    overrides = dict(overrides)
    fake = Fake(overrides.pop("backend_status", 200))
    model = ModelConfig(base_url="http://llama", **overrides)
    settings = Settings(models={"qwen-decision": model}, aliases={"jev-latest": "qwen-decision"})
    app = create_app(settings, {"qwen-decision": Backend(model, transport=httpx.MockTransport(fake))})
    with TestClient(app) as client:
        response = client.post("/api/v1/systemone", content=raw, headers={"content-type": "application/json"})
    body = response.json()
    if response.status_code == 200:
        body.pop("id")
    else:
        body = None
    return {"name": name, "config": overrides, "backend_status": fake.status, "request": raw,
            "status": response.status_code, "response": body, "calls": fake.calls if body else []}


if __name__ == "__main__":
    print(json.dumps([run(*case) for case in CASES], ensure_ascii=False, indent=1))
