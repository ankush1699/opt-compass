"""One small interface over the model providers the app can use.

anthropic: paid, official SDK. gemini: Google AI Studio REST API, which has a free tier (plain httpx,
no extra dependency). Both return the same LLMResult so generation, the judge and tracing do not care.
"""
import json
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass

import httpx

from app.config import Settings
from app.pricing import cost_usd

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_STREAM_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse"
RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 5
MAX_RETRY_WAIT_S = 120  # a longer wait means a quota (e.g. the daily free-tier cap), not a blip: fail loudly instead of sleeping for hours


@dataclass
class LLMResult:
    text: str | None
    input_tokens: int
    output_tokens: int  # includes thinking tokens, which are billed (or counted against the quota) as output
    stop_reason: str | None  # normalised: "end_turn" | "max_tokens" | "refusal" | other raw value
    cost_usd: float | None


class ProviderError(RuntimeError):
    pass


def has_credentials(provider: str, settings: Settings) -> bool:
    return bool({"anthropic": settings.anthropic_api_key, "gemini": settings.gemini_api_key}.get(provider))


def _strip_additional_properties(schema):
    """Gemini's response schema is an OpenAPI subset and does not take additionalProperties."""
    if isinstance(schema, dict):
        return {k: _strip_additional_properties(v) for k, v in schema.items() if k != "additionalProperties"}
    if isinstance(schema, list):
        return [_strip_additional_properties(v) for v in schema]
    return schema


class AnthropicProvider:
    def __init__(self, settings: Settings, model: str, effort: str = ""):
        import anthropic

        self.model, self.effort = model, effort
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def _request(self, system: str, user: str, max_tokens: int, json_schema: dict | None = None) -> dict:
        request: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        output_config: dict = {}
        if self.effort:
            output_config["effort"] = self.effort
        if json_schema:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        if output_config:
            request["output_config"] = output_config
        return request

    def _result(self, message) -> LLMResult:
        text = "".join(b.text for b in message.content if b.type == "text") or None
        if message.stop_reason == "refusal":
            text = None
        return LLMResult(
            text,
            message.usage.input_tokens,
            message.usage.output_tokens,
            message.stop_reason,
            cost_usd(self.model, message.usage.input_tokens, message.usage.output_tokens),
        )

    def complete(self, system: str, user: str, max_tokens: int, json_schema: dict | None = None) -> LLMResult:
        return self._result(self.client.messages.create(**self._request(system, user, max_tokens, json_schema)))

    def stream(self, system: str, user: str, max_tokens: int) -> Iterator[str | LLMResult]:
        """Yields text deltas as they arrive, then one final LLMResult."""
        with self.client.messages.stream(**self._request(system, user, max_tokens)) as stream:
            yield from stream.text_stream
            message = stream.get_final_message()
        yield self._result(message)


class GeminiProvider:
    _last_call = 0.0  # shared across instances so generation and judging respect one rate limit

    def __init__(self, settings: Settings, model: str, client: httpx.Client | None = None, sleep=time.sleep):
        self.model = model
        self.free = settings.gemini_free_tier
        self.min_interval = settings.gemini_min_interval_s
        self.key = settings.gemini_api_key
        self.client = client or httpx.Client(timeout=120)
        self.sleep = sleep

    def _pace(self) -> None:
        wait = GeminiProvider._last_call + self.min_interval - time.monotonic()
        if wait > 0:
            self.sleep(wait)
        GeminiProvider._last_call = time.monotonic()

    @staticmethod
    def _body(system: str, user: str, max_tokens: int, json_schema: dict | None) -> dict:
        generation_config: dict = {"maxOutputTokens": max_tokens, "temperature": 0}
        if json_schema:
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseSchema"] = _strip_additional_properties(json_schema)
        return {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": generation_config,
        }

    def complete(self, system: str, user: str, max_tokens: int, json_schema: dict | None = None) -> LLMResult:
        body = self._body(system, user, max_tokens, json_schema)
        url = GEMINI_URL.format(model=self.model)
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self._pace()
            try:
                response = self.client.post(url, json=body, headers={"x-goog-api-key": self.key})
            except httpx.TransportError as exc:  # timeouts and dropped connections are provider failures, not bugs
                raise ProviderError(f"Gemini {self.model} request failed: {type(exc).__name__}") from exc
            if response.status_code not in RETRY_STATUSES:
                break
            if attempt == MAX_ATTEMPTS:
                raise ProviderError(f"Gemini returned {response.status_code} after {MAX_ATTEMPTS} attempts")
            delay = self._retry_delay(response, attempt)
            if delay > MAX_RETRY_WAIT_S:
                raise ProviderError(f"Gemini {self.model} asked to wait {delay / 60:.0f} min (quota exhausted); not waiting. Resume later.")
            self.sleep(delay)
        if response.status_code != 200:
            raise ProviderError(f"Gemini returned {response.status_code}: {response.text[:300]}")
        return self._parse(response.json())

    def stream(self, system: str, user: str, max_tokens: int) -> Iterator[str | LLMResult]:
        """Yields text deltas from Gemini's SSE endpoint, then one final LLMResult. Retries only before the first byte."""
        body = self._body(system, user, max_tokens, None)
        url = GEMINI_STREAM_URL.format(model=self.model)
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self._pace()
            try:
                with self.client.stream("POST", url, json=body, headers={"x-goog-api-key": self.key}) as response:
                    if response.status_code in RETRY_STATUSES:
                        response.read()
                        if attempt == MAX_ATTEMPTS:
                            raise ProviderError(f"Gemini returned {response.status_code} after {MAX_ATTEMPTS} attempts")
                        delay = self._retry_delay(response, attempt)
                        if delay > MAX_RETRY_WAIT_S:
                            raise ProviderError(f"Gemini {self.model} asked to wait {delay / 60:.0f} min (quota exhausted); not waiting.")
                        self.sleep(delay)
                        continue
                    if response.status_code != 200:
                        response.read()
                        raise ProviderError(f"Gemini returned {response.status_code}: {response.text[:300]}")
                    yield from self._read_stream(response)
                    return
            except httpx.TransportError as exc:
                raise ProviderError(f"Gemini {self.model} request failed: {type(exc).__name__}") from exc

    def _read_stream(self, response: httpx.Response) -> Iterator[str | LLMResult]:
        text, usage, finish, seen = "", {}, None, False
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            chunk = json.loads(line[5:])
            seen = True
            usage = chunk.get("usageMetadata") or usage
            for candidate in (chunk.get("candidates") or [])[:1]:
                finish = candidate.get("finishReason") or finish
                for part in (candidate.get("content") or {}).get("parts") or []:
                    if part.get("text") and not part.get("thought"):
                        text += part["text"]
                        yield part["text"]
        merged = {"usageMetadata": usage}
        if seen:
            merged["candidates"] = [{"content": {"parts": [{"text": text}]}, "finishReason": finish}]
        yield self._parse(merged)

    @staticmethod
    def _retry_delay(response: httpx.Response, attempt: int) -> float:
        match = re.search(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', response.text)
        return float(match.group(1)) + 1 if match else min(60.0, 5.0 * 2**attempt)

    def _parse(self, data: dict) -> LLMResult:
        usage = data.get("usageMetadata", {})
        candidates = data.get("candidates") or []
        candidate = candidates[0] if candidates else {}
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought")) or None
        finish = candidate.get("finishReason")
        if not candidates or finish in {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "RECITATION"}:
            stop, text = "refusal", None
        else:
            stop = {"STOP": "end_turn", "MAX_TOKENS": "max_tokens"}.get(finish, finish)
        output_tokens = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
        return LLMResult(text, usage.get("promptTokenCount", 0), output_tokens, stop, 0.0 if self.free else None)


def make_provider(provider: str, settings: Settings, model: str, effort: str = ""):
    if provider == "anthropic":
        return AnthropicProvider(settings, model, effort)
    if provider == "gemini":
        return GeminiProvider(settings, model)
    raise ValueError(f"unknown provider {provider!r}")


def list_gemini_models(settings: Settings, client: httpx.Client | None = None) -> list[str]:
    """Model IDs the key can call generateContent on, so model names are looked up, never guessed."""
    client = client or httpx.Client(timeout=60)
    names: list[str] = []
    token = ""
    while True:
        response = client.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            params={"pageSize": 100, **({"pageToken": token} if token else {})},
            headers={"x-goog-api-key": settings.gemini_api_key},
        )
        if response.status_code != 200:
            raise ProviderError(f"Gemini model list returned {response.status_code}: {response.text[:200]}")
        data = response.json()
        names += [
            m["name"].removeprefix("models/")
            for m in data.get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])
        ]
        token = data.get("nextPageToken", "")
        if not token:
            return sorted(names)
