import json

import httpx
import pytest

from oversight.client import OllamaClient, OpenAICompatClient, extract_json


def test_ollama_sends_schema_seed_and_model_options(tmp_path):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"message": {"content": '{"a": 1}'}, "done_reason": "stop", "prompt_eval_count": 7, "eval_count": 3},
        )

    c = OllamaClient(tmp_path, model_options={"qwen3:8b": {"think": False}})
    c._http = httpx.Client(base_url="http://localhost:11434", transport=httpx.MockTransport(handler))
    schema = {"type": "object"}
    out = c.complete(tag="bob", model="qwen3:8b", system="s", user="u", schema=schema, max_tokens=99, sample=2)
    assert out.json() == {"a": 1}
    body = seen["body"]
    assert body["format"] == schema and body["think"] is False
    assert body["options"]["seed"] == 1002 and body["options"]["num_predict"] == 99
    assert c.usage["qwen3:8b"]["output_tokens"] == 3


def test_extract_json_handles_fences_and_chatter():
    assert json.loads(extract_json('Sure!\n```json\n{"verdict": "correct", "p_correct": 0.7}\n```')) == {
        "verdict": "correct",
        "p_correct": 0.7,
    }
    # a LaTeX-style brace before the object is skipped
    assert json.loads(extract_json('Using {x} notation: {"a": 1}')) == {"a": 1}
    assert extract_json("no json here") is None


def make_client(tmp_path, monkeypatch, handler):
    monkeypatch.setenv("TEST_KEY", "k")
    c = OpenAICompatClient(tmp_path, "https://example.test/v1", "TEST_KEY", max_retries=2)
    c._http = httpx.Client(base_url="https://example.test/v1", transport=httpx.MockTransport(handler))
    return c


def reply(content):
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


def test_schema_goes_into_prompt_and_json_is_extracted(tmp_path, monkeypatch):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return reply('Here you go: {"verdict": "incorrect", "p_correct": 0.2, "rationale": "r"}')

    c = make_client(tmp_path, monkeypatch, handler)
    schema = {"type": "object", "properties": {"verdict": {"type": "string"}}}
    out = c.complete(tag="judge:j1", model="m", system="sys", user="u", schema=schema)
    assert out.json() == {"verdict": "incorrect", "p_correct": 0.2, "rationale": "r"}
    assert "JSON Schema" in seen["body"]["messages"][0]["content"]
    assert c.usage["m"] == {"calls": 1, "input_tokens": 10, "output_tokens": 5}

    # cached: a second identical call makes no request
    seen.clear()
    c.complete(tag="judge:j1", model="m", system="sys", user="u", schema=schema)
    assert not seen


def test_retries_rate_limits(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429, json={}) if len(calls) < 3 else reply("plain text")

    c = make_client(tmp_path, monkeypatch, handler)
    assert c.complete(tag="bob", model="m", system="s", user="u").text == "plain text"
    assert len(calls) == 3


def test_client_errors_are_not_retried(tmp_path, monkeypatch):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(403, text="forbidden")

    c = make_client(tmp_path, monkeypatch, handler)
    with pytest.raises(RuntimeError, match="HTTP 403"):
        c.complete(tag="bob", model="m", system="s", user="u")
    assert len(calls) == 1


def test_identical_concurrent_calls_run_once(tmp_path):
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from oversight.client import Completion, _CachedClient

    class Slow(_CachedClient):
        provider = "slow"
        calls = 0

        def _call(self, *a, **k):
            Slow.calls += 1
            time.sleep(0.05)
            return Completion(text="x", stop_reason="stop")

    c = Slow(tmp_path)
    with ThreadPoolExecutor(8) as pool:
        outs = list(pool.map(lambda _: c.complete(tag="t", model="m", system="s", user="same"), range(8)))
    assert Slow.calls == 1 and all(o.text == "x" for o in outs)
