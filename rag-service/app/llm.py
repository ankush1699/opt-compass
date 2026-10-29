"""One small interface over the model providers the app can use.

anthropic: paid, official SDK. gemini: Google AI Studio REST API, which has a free tier (plain httpx,
no extra dependency). Both return the same LLMResult so generation, the judge and tracing do not care.
"""
import json
import re
import time
from dataclasses import dataclass

import httpx

from app.config import Settings
from app.pricing import cost_usd

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 5


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

    def complete(self, system: str, user: str, max_tokens: int, json_schema: dict | None = None) -> LLMResult:
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
        message = self.client.messages.create(**request)
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

    def complete(self, system: str, user: str, max_tokens: int, json_schema: dict | None = None) -> LLMResult:
        generation_config: dict = {"maxOutputTokens": max_tokens, "temperature": 0}
        if json_schema:
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseSchema"] = _strip_additional_properties(json_schema)
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": generation_config,
        }
        url = GEMINI_URL.format(model=self.model)
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self._pace()
            response = self.client.post(url, json=body, headers={"x-goog-api-key": self.key})
            if response.status_code not in RETRY_STATUSES:
                break
            if attempt == MAX_ATTEMPTS:
                raise ProviderError(f"Gemini returned {response.status_code} after {MAX_ATTEMPTS} attempts")
            self.sleep(self._retry_delay(response, attempt))
        if response.status_code != 200:
            raise ProviderError(f"Gemini returned {response.status_code}: {response.text[:300]}")
        return self._parse(response.json())

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
