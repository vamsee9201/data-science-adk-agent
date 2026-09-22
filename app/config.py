from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _int_env(name: str, default: int, *, minimum: int = 1, maximum: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return value


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false.")


@dataclass(frozen=True)
class Settings:
    app_env: str
    credentials_path: Path
    google_cloud_project: str | None
    google_cloud_location: str
    google_model: str
    max_upload_bytes: int
    session_ttl_seconds: int
    agent_turn_timeout_seconds: int
    max_dataset_rows: int
    max_dataset_columns: int
    max_active_sessions: int
    max_turns_per_session: int
    max_agent_turns_per_hour: int
    max_agent_turns_per_day: int
    max_concurrent_agent_turns: int
    max_model_output_tokens: int
    allowed_hosts: tuple[str, ...]
    expose_api_docs: bool

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def uses_cloud_runtime_identity(self) -> bool:
        return bool(os.getenv("K_SERVICE"))

    @property
    def vertex_configured(self) -> bool:
        # Cloud Run supplies Application Default Credentials through the service identity.
        return self.credentials_path.is_file() or self.uses_cloud_runtime_identity

    @classmethod
    def from_env(cls) -> "Settings":
        app_env = os.getenv("APP_ENV", "development").strip().lower()
        if app_env not in {"development", "test", "production"}:
            raise ValueError("APP_ENV must be development, test, or production.")
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

        default_hosts = "localhost,127.0.0.1,testserver,*.run.app"
        allowed_hosts = tuple(
            host.strip() for host in os.getenv("ALLOWED_HOSTS", default_hosts).split(",") if host.strip()
        )
        if not allowed_hosts:
            raise ValueError("ALLOWED_HOSTS must contain at least one host.")

        return cls(
            app_env=app_env,
            credentials_path=credentials,
            google_cloud_project=project,
            google_cloud_location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
            google_model=os.getenv("GOOGLE_MODEL", "gemini-3.8-flash"),
            max_upload_bytes=_int_env("MAX_UPLOAD_MB", 25, maximum=25) * 1024 * 1024,
            session_ttl_seconds=_int_env("SESSION_TTL_MINUTES", 60, maximum=240) * 60,
            agent_turn_timeout_seconds=_int_env(
                "AGENT_TURN_TIMEOUT_SECONDS", 180, maximum=300
            ),
            max_dataset_rows=_int_env("MAX_DATASET_ROWS", 100_000, maximum=500_000),
            max_dataset_columns=_int_env("MAX_DATASET_COLUMNS", 200, maximum=1_000),
            max_active_sessions=_int_env("MAX_ACTIVE_SESSIONS", 25, maximum=500),
            max_turns_per_session=_int_env("MAX_TURNS_PER_SESSION", 20, maximum=1_000),
            max_agent_turns_per_hour=_int_env("MAX_AGENT_TURNS_PER_HOUR", 30, maximum=10_000),
            max_agent_turns_per_day=_int_env("MAX_AGENT_TURNS_PER_DAY", 100, maximum=100_000),
            max_concurrent_agent_turns=_int_env(
                "MAX_CONCURRENT_AGENT_TURNS", 1, maximum=20
            ),
            max_model_output_tokens=_int_env("MAX_MODEL_OUTPUT_TOKENS", 2_048, maximum=8_192),
            allowed_hosts=allowed_hosts,
            expose_api_docs=_bool_env("EXPOSE_API_DOCS", app_env != "production"),
        )


settings = Settings.from_env()
