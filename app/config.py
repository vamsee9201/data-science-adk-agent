from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    credentials_path: Path
    google_cloud_project: str | None
    google_cloud_location: str
    google_model: str
    max_upload_bytes: int
    session_ttl_seconds: int
    agent_turn_timeout_seconds: int

    @classmethod
    def from_env(cls) -> "Settings":
        credentials = Path(
            os.getenv(
                "GOOGLE_APPLICATION_CREDENTIALS",
                ROOT / ".secrets" / "vertex-service-account.json",
            )
        ).expanduser()
        if not credentials.is_absolute():
            credentials = ROOT / credentials

        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project and credentials.is_file():
            try:
                project = json.loads(credentials.read_text(encoding="utf-8")).get("project_id")
            except (OSError, json.JSONDecodeError):
                project = None

        return cls(
            credentials_path=credentials,
            google_cloud_project=project,
            google_cloud_location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
            google_model=os.getenv("GOOGLE_MODEL", "gemini-3.8-flash"),
            max_upload_bytes=int(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024,
            session_ttl_seconds=int(os.getenv("SESSION_TTL_MINUTES", "60")) * 60,
            agent_turn_timeout_seconds=int(os.getenv("AGENT_TURN_TIMEOUT_SECONDS", "180")),
        )


settings = Settings.from_env()
