from __future__ import annotations

import threading
import time
from collections import deque

from app.config import settings
from app.sessions import WorkspaceSession


class UsageLimitExceeded(RuntimeError):
    def __init__(self, message: str, retry_after: int = 60) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class UsageGuard:
    """Small in-process guardrail for model calls.

    Deploying one Cloud Run instance makes these limits useful cost brakes. They are not a
    billing hard cap because counters reset when the container restarts.
    """

    def __init__(
        self,
        *,
        hourly_limit: int,
        daily_limit: int,
        concurrent_limit: int,
        per_session_limit: int,
    ) -> None:
        self.hourly_limit = hourly_limit
        self.daily_limit = daily_limit
        self.concurrent_limit = concurrent_limit
        self.per_session_limit = per_session_limit
        self._turns: deque[float] = deque()
        self._active = 0
        self._lock = threading.Lock()

    def acquire(self, session: WorkspaceSession) -> None:
        now = time.time()
        with self._lock:
            while self._turns and self._turns[0] <= now - 86_400:
                self._turns.popleft()
            hourly_count = sum(timestamp > now - 3_600 for timestamp in self._turns)
            if session.agent_turns >= self.per_session_limit:
                raise UsageLimitExceeded(
                    "This session has reached its analysis-turn limit. Start a new session to continue.",
                    retry_after=60,
                )
            if len(self._turns) >= self.daily_limit:
                raise UsageLimitExceeded(
                    "The daily analysis limit has been reached. Please try again tomorrow.",
                    retry_after=3_600,
                )
            if hourly_count >= self.hourly_limit:
                raise UsageLimitExceeded(
                    "The hourly analysis limit has been reached. Please try again later.",
                    retry_after=300,
                )
            if self._active >= self.concurrent_limit:
                raise UsageLimitExceeded(
                    "The analyst is busy with another request. Please retry shortly.",
                    retry_after=10,
                )
            self._turns.append(now)
            self._active += 1
            session.agent_turns += 1

    def release(self) -> None:
        with self._lock:
            self._active = max(0, self._active - 1)

    def reset(self) -> None:
        with self._lock:
            self._turns.clear()
            self._active = 0


usage_guard = UsageGuard(
    hourly_limit=settings.max_agent_turns_per_hour,
    daily_limit=settings.max_agent_turns_per_day,
    concurrent_limit=settings.max_concurrent_agent_turns,
    per_session_limit=settings.max_turns_per_session,
)
