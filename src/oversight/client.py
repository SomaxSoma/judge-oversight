"""LLM access: a disk-cached Anthropic client and a deterministic mock for dry runs.

Every call is keyed by (model, system, user, schema, effort, max_tokens, sample), plus `extra`
when one is given, so re-running the pipeline resumes from the cache instead of paying twice.
"""

import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class Completion:
    text: str | None  # None when the model refused
    stop_reason: str
    input_tokens: int = 0
    output_tokens: int = 0

    def json(self) -> dict | None:
        if self.text is None or self.stop_reason == "refusal":
            return None
        try:
            return json.loads(self.text)
        except json.JSONDecodeError:
            return None


class Client(Protocol):
    def complete(
        self,
        *,
        tag: str,
        model: str,
        system: str,
        user: str,
        schema: dict | None = None,
        effort: str | None = None,
        max_tokens: int = 8000,
        sample: int = 0,
        extra: dict | None = None,
    ) -> Completion: ...


def _cache_key(**fields) -> str:
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


class _CachedClient:
    """Disk cache + usage accounting shared by the real backends; subclasses implement _call."""

    provider = ""
    supports_extra = False  # per-call backend options, e.g. Ollama's {"think": True}

    def __init__(self, cache_dir: str | Path):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.usage: dict[str, dict[str, int]] = {}

    def complete(self, *, tag, model, system, user, schema=None, effort=None, max_tokens=8000, sample=0, extra=None):
        if extra and not self.supports_extra:
            raise ValueError(f"{type(self).__name__} does not take per-call options: {extra}")
        fields = dict(
            provider=self.provider, model=model, system=system, user=user, schema=schema,
            effort=effort, max_tokens=max_tokens, sample=sample,
        )
        if extra:  # only when set, so keys of calls without it stay as they were
            fields["extra"] = extra
        key = _cache_key(**fields)
        path = self._cache_dir / key[:2] / f"{key}.json"
        # One lock per key: identical concurrent calls run once, and on Windows a reader never
        # races os.replace on the same file (that raises PermissionError).
        with self._key_lock(key):
            if path.exists():
                return Completion(**json.loads(path.read_text(encoding="utf-8")))

            result = self._call(model, system, user, schema, effort, max_tokens, sample, extra)
            self._record_usage(model, result)

            path.parent.mkdir(exist_ok=True)
            tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps(asdict(result)), encoding="utf-8")
            os.replace(tmp, path)
            return result

    def _key_lock(self, key: str) -> threading.Lock:
        with self._lock:
            locks = self.__dict__.setdefault("_key_locks", {})
            return locks.setdefault(key, threading.Lock())

    def _call(self, model, system, user, schema, effort, max_tokens, sample, extra=None) -> Completion:
        raise NotImplementedError

    def _record_usage(self, model: str, result: Completion) -> None:
        with self._lock:
            u = self.usage.setdefault(model, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
            u["calls"] += 1
            u["input_tokens"] += result.input_tokens
            u["output_tokens"] += result.output_tokens


class AnthropicClient(_CachedClient):
    provider = "anthropic"

    def __init__(self, cache_dir: str | Path, max_retries: int = 6):
        import anthropic

        super().__init__(cache_dir)
        self._client = anthropic.Anthropic(max_retries=max_retries)

    def _call(self, model, system, user, schema, effort, max_tokens, sample, extra=None):
        kwargs = dict(model=model, max_tokens=max_tokens, system=system, messages=[{"role": "user", "content": user}])
        output_config = {}
        if effort:
            output_config["effort"] = effort
        if schema:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        if output_config:
            kwargs["output_config"] = output_config

        resp = self._client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if b.type == "text")
        return Completion(
            text=None if resp.stop_reason == "refusal" else text,
            stop_reason=resp.stop_reason,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )


class OpenAICompatClient(_CachedClient):
    """Any OpenAI-compatible chat-completions endpoint, e.g. OpenCode's free models.

    Free providers differ in JSON-schema support, so the schema goes into the system prompt and the
    first JSON object in the reply is extracted. `effort` is ignored. 429s and 5xx are retried.
    """

    provider = "openai_compat"

    def __init__(self, cache_dir, base_url: str, api_key_env: str, timeout: float = 600, max_retries: int = 6):
        import httpx

        super().__init__(cache_dir)
        key = os.environ.get(api_key_env)
        if not key:
            raise SystemExit(f"{api_key_env} is not set (put it in .env).")
        self.provider = f"openai_compat:{base_url}"
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout, headers={"Authorization": f"Bearer {key}"}
        )
        self._max_retries = max_retries

    def _call(self, model, system, user, schema, effort, max_tokens, sample, extra=None):
        if schema:
            system += (
                "\n\nReply with only a JSON object, no other text, matching this JSON Schema:\n" + json.dumps(schema)
            )
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "seed": 1000 + sample,
        }
        data = self._post_with_retries(body)
        choice = data["choices"][0]
        text = choice["message"].get("content") or ""
        if schema:
            text = extract_json(text) or text
        usage = data.get("usage") or {}
        return Completion(
            text=text,
            stop_reason=choice.get("finish_reason") or "stop",
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )

    def _post_with_retries(self, body: dict) -> dict:
        import httpx

        for attempt in range(self._max_retries + 1):
            try:
                resp = self._http.post("/chat/completions", json=body)
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt == self._max_retries:
                    raise
            else:
                if resp.status_code < 400:
                    return resp.json()
                if resp.status_code != 429 and resp.status_code < 500 or attempt == self._max_retries:
                    raise RuntimeError(f"{body['model']}: HTTP {resp.status_code}: {resp.text[:500]}")
                retry_after = resp.headers.get("retry-after")
                if retry_after and retry_after.isdigit():
                    time.sleep(min(int(retry_after), 120))
                    continue
            time.sleep(min(2**attempt * 2, 120))
        raise AssertionError("unreachable")


class OllamaClient(_CachedClient):
    """Local open-weights models served by Ollama: free, unlimited, and nothing leaves the machine.

    Ollama enforces the JSON schema itself. `effort` is ignored. Each sample gets its own seed, so
    samples differ but reruns reproduce. `model_options` maps a model name to extra request
    fields, e.g. {"qwen3:8b": {"think": False}}; a call's `extra` overrides them for that call.
    """

    provider = "ollama"
    supports_extra = True

    def __init__(self, cache_dir, base_url="http://localhost:11434", num_ctx=8192, model_options=None, timeout=1800):
        import httpx

        super().__init__(cache_dir)
        self._http = httpx.Client(base_url=base_url, timeout=timeout)
        self._num_ctx = num_ctx
        self._model_options = model_options or {}

    def _call(self, model, system, user, schema, effort, max_tokens, sample, extra=None):
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {"num_ctx": self._num_ctx, "num_predict": max_tokens, "seed": 1000 + sample},
            **self._model_options.get(model, {}),
            **(extra or {}),
        }
        if schema:
            body["format"] = schema
        resp = self._http.post("/api/chat", json=body)
        if resp.status_code >= 400:
            raise RuntimeError(f"{model}: HTTP {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        return Completion(
            text=data["message"]["content"],
            stop_reason=data.get("done_reason", "stop"),
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
        )


def extract_json(text: str) -> str | None:
    """Return the first parseable JSON object in text (models may add fences or chatter)."""
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(text, i)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return json.dumps(obj)
    return None


class MockClient:
    """Deterministic stand-in used by tests and `--mock` dry runs. Makes no network calls."""

    def __init__(self):
        self.usage: dict[str, dict[str, int]] = {}
        self.calls: list[str] = []

    @staticmethod
    def _h(*parts) -> int:
        return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest(), 16)

    def complete(self, *, tag, model, system, user, schema=None, effort=None, max_tokens=8000, sample=0, extra=None):
        self.calls.append(tag)
        h = self._h(tag, model, user, sample)
        if tag == "alice":
            out = {"explanation": "Mock reasoning about the question.", "final_option": "ABCD"[h % 4]}
        elif tag == "bob":
            return Completion(text="Check the step that sets up the governing equation; if it is wrong, the conclusion changes.", stop_reason="end_turn")
        elif tag == "leak_check":
            out = {"reason": "mock", "states_verdict": False, "states_answer": False}
        elif tag == "bob_solve":
            out = {"reasoning": f"mock working {sample}", "final_option": "ABCD"[h % 4]}
        elif tag == "leak_audit":
            out = {"rationale": "mock", "prediction": "correct" if h % 2 else "incorrect", "p_correct": (h % 100) / 100}
        elif tag.startswith("judge"):
            out = {"rationale": "mock", "verdict": "correct" if h % 2 else "incorrect", "p_correct": (h % 100) / 100}
        elif tag.startswith("checks_write:"):
            out = {"checks": [{"question": f"Mock check {i} ({tag})?", "alice_value": f"value {i}"} for i in range(3)]}
        elif tag == "check_answer":
            out = {"answer": f"mock answer {h % 5}", "confidence": (h % 100) / 100}
        elif tag.startswith("checks_final:"):
            out = {"agreements": [{"check": i, "p_agree": ((h >> i) % 100) / 100} for i in (1, 2, 3)],
                   "rationale": "mock", "verdict": "correct" if h % 2 else "incorrect", "p_correct": (h % 100) / 100}
        elif tag == "leak_check_checks":
            out = {"reason": "mock", "states_verdict": False, "states_answer": False, "ungrounded": False}
        else:
            raise ValueError(f"MockClient has no behaviour for tag {tag!r}")
        return Completion(text=json.dumps(out), stop_reason="end_turn")
