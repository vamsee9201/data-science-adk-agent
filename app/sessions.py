from __future__ import annotations

import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import settings


class SessionNotFound(KeyError):
    pass


class SessionCapacityExceeded(RuntimeError):
    pass


@dataclass
class TransformationProposal:
    id: str
    operation: str
    dataframe: pd.DataFrame
    summary: dict


@dataclass
class WorkspaceSession:
    id: str
    directory: Path
    created_at: float = field(default_factory=time.time)
    touched_at: float = field(default_factory=time.time)
    dataset_name: str | None = None
    dataframe: pd.DataFrame | None = None
    original_dataframe: pd.DataFrame | None = None
    chat_history: list[dict] = field(default_factory=list)
    proposal: TransformationProposal | None = None
    artifacts: dict[str, Path] = field(default_factory=dict)
    model_results: list[dict] = field(default_factory=list)
    busy: bool = False
    cancel_requested: bool = False
    agent_turns: int = 0
    event_sink: Any = None

    def touch(self) -> None:
        self.touched_at = time.time()

    def set_dataset(self, name: str, dataframe: pd.DataFrame) -> None:
        for path in self.artifacts.values():
            path.unlink(missing_ok=True)
        self.artifacts.clear()
        self.dataset_name = name
        self.original_dataframe = dataframe.copy(deep=True)
        self.dataframe = dataframe.copy(deep=True)
        self.proposal = None
        self.model_results.clear()
        self.chat_history.clear()
        self.touch()

    def require_dataframe(self) -> pd.DataFrame:
        if self.dataframe is None:
            raise ValueError("No dataset is active. Upload a CSV or choose a sample dataset first.")
        return self.dataframe

    def add_artifact(self, path: Path) -> str:
        artifact_id = uuid.uuid4().hex
        self.artifacts[artifact_id] = path
        self.touch()
        return artifact_id


class SessionRegistry:
    def __init__(self, ttl_seconds: int = 3600, max_sessions: int = 25) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self._sessions: dict[str, WorkspaceSession] = {}
        self._lock = threading.RLock()

    def create(self, *, replacing_session_id: str | None = None) -> WorkspaceSession:
        with self._lock:
            self._cleanup_expired_unlocked(time.time())
            replacing_existing = replacing_session_id in self._sessions
            if len(self._sessions) >= self.max_sessions and not replacing_existing:
                raise SessionCapacityExceeded(
                    "The workspace is at capacity. Please retry after an inactive session expires."
                )
            session_id = str(uuid.uuid4())
            directory = Path(tempfile.mkdtemp(prefix=f"ds-agent-{session_id[:8]}-"))
            session = WorkspaceSession(id=session_id, directory=directory)
            self._sessions[session_id] = session
            return session

    def get(self, session_id: str, *, touch: bool = True) -> WorkspaceSession:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionNotFound(session_id)
            if time.time() - session.touched_at > self.ttl_seconds:
                self._delete_unlocked(session_id)
                raise SessionNotFound(session_id)
            if touch:
                session.touch()
            return session

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._delete_unlocked(session_id)

    def _delete_unlocked(self, session_id: str) -> bool:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        shutil.rmtree(session.directory, ignore_errors=True)
        return True

    def cleanup_expired(self) -> int:
        now = time.time()
        with self._lock:
            return self._cleanup_expired_unlocked(now)

    def _cleanup_expired_unlocked(self, now: float) -> int:
        expired = [
                session_id
                for session_id, session in self._sessions.items()
                if now - session.touched_at > self.ttl_seconds
            ]
        for session_id in expired:
            self._delete_unlocked(session_id)
        return len(expired)

    def close(self) -> None:
        with self._lock:
            for session_id in list(self._sessions):
                self._delete_unlocked(session_id)


registry = SessionRegistry(settings.session_ttl_seconds, settings.max_active_sessions)
