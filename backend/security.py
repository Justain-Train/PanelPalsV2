"""API-key authentication and per-client rate limiting."""

import hashlib
import hmac
import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional

from fastapi import Depends, Header, HTTPException, Request, status

from backend.config import settings

ANONYMOUS = "anonymous"


def require_api_key(x_api_key: Optional[str] = Header(None)) -> str:
    """Check X-API-Key. Without configured keys, only DEBUG mode is allowed through."""
    keys = [k.get_secret_value() for k in settings.API_KEYS if k.get_secret_value()]
    if not keys:
        if settings.DEBUG:
            return ANONYMOUS
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Server has no API keys configured")
    # Constant-time comparison
    if not x_api_key or not any(hmac.compare_digest(x_api_key.encode(), k.encode()) for k in keys):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid or missing API key",
                            headers={"WWW-Authenticate": "ApiKey"})
    return x_api_key


class RateLimiter:
    """Sliding one-minute window per client. In-memory, so per worker process."""

    WINDOW_SECONDS = 60.0

    def __init__(self):
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, client_id: str, limit: int) -> None:
        """Record a request; raise 429 past `limit` per minute. limit <= 0 disables."""
        if limit <= 0:
            return
        now = time.monotonic()
        with self._lock:
            hits = self._hits[client_id]
            while hits and now - hits[0] >= self.WINDOW_SECONDS:
                hits.popleft()
            if len(hits) >= limit:
                retry_after = max(1, int(self.WINDOW_SECONDS - (now - hits[0])) + 1)
                raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                    detail="Rate limit exceeded",
                                    headers={"Retry-After": str(retry_after)})
            hits.append(now)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


rate_limiter = RateLimiter()


def _client_id(request: Request, api_key: str) -> str:
    if api_key != ANONYMOUS:
        return "key:" + hashlib.sha256(api_key.encode()).hexdigest()[:16]
    return "ip:" + (request.client.host if request.client else "unknown")


def enforce_request_limits(request: Request, api_key: str = Depends(require_api_key)) -> None:
    """Route dependency: authenticate, then rate limit."""
    rate_limiter.check(_client_id(request, api_key), settings.RATE_LIMIT_PER_MINUTE)
