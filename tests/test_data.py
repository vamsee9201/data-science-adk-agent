from __future__ import annotations

import os
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.analytics import (
    aggregate_data,
    apply_proposal,
    correlation_analysis,
    create_chart,
    propose_cleaning,
    reset_dataset,
    sort_data,
    statistical_test,
    train_baseline_model,
)
from app.data import DatasetValidationError, load_sample, parse_csv, profile_dataset
from app.sessions import SessionRegistry


def test_parse_csv_and_profile() -> None:
    frame = parse_csv(b"name,value\na,1\nb,\n", 1_000)
    assert list(frame.columns) == ["name", "value"]
    profile = profile_dataset(frame)
    assert profile["shape"]["rows"] == 2
    assert profile["columns"]["value"]["missing"] == 1


def test_analyst_plan_guard_blocks_tools_until_plan() -> None:
    from app.agent import _require_analysis_plan, _start_analyst_turn, tool_activity_label

    context = SimpleNamespace(state={"analysis_plan_recorded": True})
    _start_analyst_turn(context)
    assert context.state["analysis_plan_recorded"] is False
    blocked = _require_analysis_plan(SimpleNamespace(name="inspect_dataset"), {}, context)
    assert blocked and "not executed" in blocked["error"]
    assert _require_analysis_plan(SimpleNamespace(name="record_analysis_plan"), {}, context) is None
    assert context.state["analysis_plan_recorded"] is True
    assert _require_analysis_plan(SimpleNamespace(name="inspect_dataset"), {}, context) is None
    assert tool_activity_label("analyze_correlations") == "Evaluating relationships"
    assert tool_activity_label("unknown_internal_tool") == "Using an analysis tool"


@pytest.mark.parametrize(
    "content, message",
    [
        (b"", "empty"),
        ("name\nJosé".encode("latin-1"), "UTF-8"),
        (b"a,a\n1,2\n", "Duplicate"),
        (b"a,b\n", "no data rows"),
    ],
)
def test_parse_csv_rejects_invalid_files(content: bytes, message: str) -> None:
    with pytest.raises(DatasetValidationError, match=message):
        parse_csv(content, 1_000)


def test_parse_csv_enforces_processing_dimensions() -> None:
    with pytest.raises(DatasetValidationError, match="columns; the limit is 2"):
        parse_csv(b"a,b,c\n1,2,3\n", 1_000, max_columns=2)
    with pytest.raises(DatasetValidationError, match="more than 2 rows"):
        parse_csv(b"a\n1\n2\n3\n", 1_000, max_rows=2)


def test_usage_guard_limits_and_releases() -> None:
    from app.guardrails import UsageGuard, UsageLimitExceeded

    local_registry = SessionRegistry(max_sessions=2)
    first = local_registry.create()
    second = local_registry.create()
    guard = UsageGuard(hourly_limit=2, daily_limit=3, concurrent_limit=1, per_session_limit=1)
    guard.acquire(first)
    with pytest.raises(UsageLimitExceeded, match="busy"):
        guard.acquire(second)
    guard.release()
    with pytest.raises(UsageLimitExceeded, match="session"):
        guard.acquire(first)
    guard.acquire(second)
    guard.release()
    local_registry.close()


def test_session_registry_capacity_and_expiry() -> None:
    from app.sessions import SessionCapacityExceeded

    local_registry = SessionRegistry(ttl_seconds=1, max_sessions=1)
    first = local_registry.create()
    with pytest.raises(SessionCapacityExceeded, match="capacity"):
        local_registry.create()
    first.touched_at = time.time() - 2
    replacement = local_registry.create()
    assert replacement.id != first.id
    local_registry.close()


def test_aggregate_correlation_and_statistics() -> None:
    frame = pd.DataFrame(
        {"group": ["a", "a", "b", "b"], "x": [1, 2, 4, 5], "y": [2, 4, 8, 10]}
    )
    aggregate = aggregate_data(frame, "group", "x", "mean")
    assert aggregate["rows"][0]["mean"] == 4.5
    assert correlation_analysis(frame)["pairs"][0]["correlation"] == 1.0
    assert sort_data(frame, "x", descending=True, limit=1)["rows"][0]["x"] == 5
    result = statistical_test(frame, "x", "y")
    assert result["test"] == "Pearson correlation"
    assert result["p_value"] < 0.01


def test_cleaning_proposal_requires_apply_and_can_reset() -> None:
    local_registry = SessionRegistry()
    session = local_registry.create()
    original = pd.DataFrame({"x": [1.0, np.nan, 3.0], "label": ["a", "a", "b"]})
    session.set_dataset("test", original)
    proposal = propose_cleaning(session, "fill_missing", ["x"], "median")
    assert session.dataframe["x"].isna().sum() == 1
    apply_proposal(session, proposal["proposal_id"])
    assert session.dataframe["x"].isna().sum() == 0
    reset_dataset(session)
    assert session.dataframe["x"].isna().sum() == 1
    local_registry.close()


def test_chart_is_scoped_to_session() -> None:
    local_registry = SessionRegistry()
    session = local_registry.create()
    session.set_dataset("test", pd.DataFrame({"x": [1, 2, 3, 4]}))
    chart = create_chart(session, "histogram", x="x")
    assert chart["artifact_id"] in session.artifacts
    assert session.artifacts[chart["artifact_id"]].is_file()
    directory = session.directory
    local_registry.close()
    assert not directory.exists()


@pytest.mark.parametrize("sample_id,target,problem", [("iris", "target", "classification"), ("diabetes", "target", "regression")])
def test_baseline_model(sample_id: str, target: str, problem: str) -> None:
    local_registry = SessionRegistry()
    session = local_registry.create()
    name, frame = load_sample(sample_id)
    session.set_dataset(name, frame)
    result = train_baseline_model(session, target)
    assert result["problem_type"] == problem
    assert result["selected_model"] != "dummy"
    assert result["chart"]["artifact_id"] in session.artifacts
    if "target_label" in frame:
        assert "target_label" in result["excluded_columns"]
    local_registry.close()


@pytest.mark.vertex
@pytest.mark.skipif(os.getenv("RUN_VERTEX_SMOKE") != "1", reason="opt-in Vertex smoke test")
@pytest.mark.asyncio
async def test_vertex_smoke() -> None:
    from app.agent import agent_runtime
    from app.sessions import registry

    session = registry.create()
    _, frame = load_sample("iris")
    session.set_dataset("Iris", frame)
    await agent_runtime.create_session(session.id)
    events = [event async for event in agent_runtime.stream(session.id, "How many rows are there?")]
    assert any(event["event"] == "message" for event in events)
    await agent_runtime.delete_session(session.id)
    registry.delete(session.id)


@pytest.mark.vertex
@pytest.mark.skipif(os.getenv("RUN_VERTEX_SMOKE") != "1", reason="opt-in Vertex smoke test")
@pytest.mark.asyncio
async def test_vertex_delegated_analysis_smoke() -> None:
    from app.agent import agent_runtime
    from app.sessions import registry

    session = registry.create()
    _, frame = load_sample("iris")
    session.set_dataset("Iris", frame)
    await agent_runtime.create_session(session.id)
    events = [
        event
        async for event in agent_runtime.stream(
            session.id,
            "Perform a broad exploratory analysis of this dataset. Inspect it, analyze missing "
            "values and correlations, then summarize the strongest findings.",
        )
    ]
    event_types = [event["event"] for event in events]
    assert "analysis_plan" in event_types
    assert "message" in event_types
    await agent_runtime.delete_session(session.id)
    registry.delete(session.id)
