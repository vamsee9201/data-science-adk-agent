from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from app.agent import agent_runtime
from app.analytics import create_chart, propose_cleaning
from app.main import app
from app.sessions import registry


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def create_session(client: TestClient) -> str:
    response = client.post("/api/sessions")
    assert response.status_code == 201
    return response.json()["session_id"]


def test_sample_dataset_lifecycle(client: TestClient) -> None:
    session_id = create_session(client)
    response = client.post(
        f"/api/sessions/{session_id}/datasets/sample", json={"sample_id": "iris"}
    )
    assert response.status_code == 200
    assert response.json()["rows"] == 150

    blocked = client.post(
        f"/api/sessions/{session_id}/datasets/sample", json={"sample_id": "wine"}
    )
    assert blocked.status_code == 409
    replaced = client.post(
        f"/api/sessions/{session_id}/datasets/sample",
        json={"sample_id": "wine", "confirm_replace": True},
    )
    assert replaced.status_code == 200
    assert replaced.json()["name"] == "Wine"
    assert client.delete(f"/api/sessions/{session_id}").json() == {"deleted": True}
    assert client.get(f"/api/sessions/{session_id}/dataset").status_code == 404


def test_session_replacement_is_atomic_and_retires_previous(client: TestClient) -> None:
    previous = create_session(client)
    client.post(f"/api/sessions/{previous}/datasets/sample", json={"sample_id": "iris"})

    response = client.post(
        "/api/sessions/replace", json={"previous_session_id": previous}
    )
    assert response.status_code == 201
    payload = response.json()
    assert payload["session_id"] != previous
    assert payload["usage"] == {"used": 0, "limit": 20, "remaining": 20}
    assert client.get(f"/api/sessions/{previous}/dataset").status_code == 404
    assert client.get(f"/api/sessions/{payload['session_id']}/dataset").status_code == 400


def test_upload_validation_and_isolation(client: TestClient) -> None:
    first, second = create_session(client), create_session(client)
    response = client.post(
        f"/api/sessions/{first}/datasets/upload",
        files={"file": ("small.csv", b"category,value\na,1\nb,2\n", "text/csv")},
    )
    assert response.status_code == 200
    assert client.get(f"/api/sessions/{second}/dataset").status_code == 400
    duplicate = client.post(
        f"/api/sessions/{second}/datasets/upload",
        files={"file": ("bad.csv", b"a,a\n1,2\n", "text/csv")},
    )
    assert duplicate.status_code == 422


def test_streamed_chat_and_busy_reset(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    session_id = create_session(client)
    client.post(f"/api/sessions/{session_id}/datasets/sample", json={"sample_id": "iris"})

    async def fake_stream(_, message: str):
        yield {"event": "analysis_plan", "steps": ["Inspect", "Summarize"]}
        yield {"event": "message", "message": f"Completed: {message}"}

    monkeypatch.setattr(agent_runtime, "stream", fake_stream)
    response = client.post(
        f"/api/sessions/{session_id}/chat", json={"message": "Analyze the dataset"}
    )
    assert response.status_code == 200
    lines = response.text.strip().splitlines()
    assert '"event": "analysis_plan"' in lines[0]
    assert '"event": "message"' in lines[1]
    assert '"event": "turn_complete"' in lines[2]
    assert '"remaining": 19' in lines[2]
    assert registry.get(session_id).busy is False


def test_explicit_modeling_mismatch_is_blocked_before_agent_turn(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = create_session(client)
    client.post(f"/api/sessions/{session_id}/datasets/sample", json={"sample_id": "iris"})

    async def should_not_run(*_):
        raise AssertionError("agent should not run for a deterministic target mismatch")
        yield

    monkeypatch.setattr(agent_runtime, "stream", should_not_run)
    response = client.post(
        f"/api/sessions/{session_id}/chat",
        json={"message": "Build a baseline regression model using target."},
    )
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.strip().splitlines()]
    assert [event["event"] for event in events] == ["warning", "message", "turn_complete"]
    assert events[-1]["status"] == "blocked"
    assert events[-1]["usage"]["remaining"] == 20


def test_expired_session_is_deleted() -> None:
    session = registry.create()
    directory = session.directory
    session.touched_at = time.time() - registry.ttl_seconds - 1
    assert registry.cleanup_expired() >= 1
    assert not directory.exists()


def test_transformation_apply_reject_and_reset(client: TestClient) -> None:
    session_id = create_session(client)
    client.post(f"/api/sessions/{session_id}/datasets/sample", json={"sample_id": "iris"})
    session = registry.get(session_id)
    session.dataframe.loc[0, "sepal length (cm)"] = None
    proposal = propose_cleaning(session, "fill_missing", ["sepal length (cm)"], "median")
    applied = client.post(
        f"/api/sessions/{session_id}/transformations/{proposal['proposal_id']}/apply"
    )
    assert applied.status_code == 200
    assert applied.json()["dataset"]["rows"] == 150
    assert session.dataframe["sepal length (cm)"].isna().sum() == 0
    reset = client.post(f"/api/sessions/{session_id}/dataset/reset")
    assert reset.status_code == 200
    assert reset.json()["rows"] == 150

    proposal = propose_cleaning(session, "drop_duplicates")
    rejected = client.delete(
        f"/api/sessions/{session_id}/transformations/{proposal['proposal_id']}"
    )
    assert rejected.json() == {"rejected": True}
    assert session.proposal is None


def test_artifacts_are_session_scoped(client: TestClient) -> None:
    first, second = create_session(client), create_session(client)
    client.post(f"/api/sessions/{first}/datasets/sample", json={"sample_id": "iris"})
    chart = create_chart(registry.get(first), "histogram", x="sepal length (cm)")
    assert client.get(chart["url"]).status_code == 200
    cross_session_url = chart["url"].replace(first, second)
    assert client.get(cross_session_url).status_code == 404


def test_busy_and_cancel_contract(client: TestClient) -> None:
    session_id = create_session(client)
    client.post(f"/api/sessions/{session_id}/datasets/sample", json={"sample_id": "iris"})
    session = registry.get(session_id)
    assert client.post(f"/api/sessions/{session_id}/chat/cancel").status_code == 409
    session.busy = True
    busy = client.post(f"/api/sessions/{session_id}/chat", json={"message": "Overview"})
    assert busy.status_code == 409
    cancel = client.post(f"/api/sessions/{session_id}/chat/cancel")
    assert cancel.status_code == 200
    assert session.cancel_requested is True
    session.busy = False


def test_static_app_is_served(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Data Science AI Agent" in response.text
    assert "Built with Google Agent Development Kit" in response.text
    assert 'id="usage-status"' in response.text
    assert response.headers["x-frame-options"] == "DENY"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]

    script = client.get("/static/app.js?v=5")
    assert script.status_code == 200
    assert 'sessionStorage.getItem("dataScienceAgentSessionId")' in script.text
    assert 'api("/api/sessions/replace"' in script.text


def test_api_responses_are_not_cached(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
