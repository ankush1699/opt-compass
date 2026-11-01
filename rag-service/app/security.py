"""API-key auth and a per-key rate limit for the gateway."""
import hmac
import threading
import time
from collections import deque

from fastapi import HTTPException, Request


def parse_keys(raw: str) -> dict[str, str]:
    """"alice:secret1,bob:secret2" -> {"alice": "secret1", "bob": "secret2"}. Entries without a name or secret are ignored."""
    keys: dict[str, str] = {}
    for entry in raw.split(","):
        name, _, secret = entry.strip().partition(":")
        if name.strip() and secret.strip():
            keys[name.strip()] = secret.strip()
    return keys


def presented_key(request: Request) -> str:
    bearer = request.headers.get("authorization", "")
    if bearer.lower().startswith("bearer "):
        return bearer[7:].strip()
    return request.headers.get("x-api-key", "").strip()


def authenticate(keys: dict[str, str], presented: str) -> str | None:
    """Name of the key that matches, or None. Compares against every key so timing does not reveal which one was close."""
    found = None
    for name, secret in keys.items():
        if hmac.compare_digest(secret.encode(), presented.encode()):
            found = name
    return found


class RateLimiter:
    """Sliding window per key name, in memory: one process only, and counts reset on restart."""

    def __init__(self, per_minute: int, window_s: float = 60.0, clock=time.monotonic):
        self.limit, self.window, self.clock = per_minute, window_s, clock
        self.hits: dict[str, deque[float]] = {}
        self.lock = threading.Lock()

    def check(self, name: str) -> float | None:
        """None if the request is allowed (and counted); otherwise seconds until the oldest request leaves the window."""
        now = self.clock()
        with self.lock:
            hits = self.hits.setdefault(name, deque())
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return self.window - (now - hits[0])
            hits.append(now)
            return None


def require_key(request: Request) -> str:
    """FastAPI dependency: returns the caller's key name, or raises 503 (no keys configured), 401 or 429."""
    state = request.app.state
    if not state.api_keys:
        raise HTTPException(status_code=503, detail="No API keys are configured on the server.")
    name = authenticate(state.api_keys, presented_key(request))
    if name is None:
        raise HTTPException(status_code=401, detail="Missing or invalid API key.", headers={"WWW-Authenticate": "Bearer"})
    wait = state.limiter.check(name)
    if wait is not None:
        raise HTTPException(
            status_code=429, detail="Rate limit exceeded.", headers={"Retry-After": str(max(1, int(wait + 0.999)))}
        )
    request.state.key_name = name
    return name
