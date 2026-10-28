"""确定性对话推荐流程的约束、trace 和回退。"""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from recagent.agent.policy import parse_preferences
from recagent.agent.runtime import DeterministicAgent
from recagent.agent.tools import RecommendationTools
from recagent.engine.baseline import Recommendation
from recagent.engine.store import Catalog, MovieInfo


@pytest.fixture
def tools():
    catalog = Catalog(
        movies={
            1: MovieInfo(1, "One", ("Sci-Fi",)),
            2: MovieInfo(2, "Two", ("Sci-Fi", "Comedy")),
            3: MovieInfo(3, "Three", ("Horror",)),
            4: MovieInfo(4, "Four", ("Sci-Fi",)),
        },
        history={1: (1,), 2: (1, 2), 3: (2, 4), 4: (3,)},
        item_users={
            1: frozenset({1, 2}),
            2: frozenset({2, 3}),
            3: frozenset({4}),
            4: frozenset({3}),
        },
        popularity={1: 2, 2: 2, 3: 1, 4: 1},
    )
    return RecommendationTools(catalog)


def test_parse_explicit_and_unparsed_constraints():
    prefs = parse_preferences("想看科幻，不要恐怖，也不要太压抑")
    assert prefs.genres == ("Sci-Fi",)
    assert prefs.exclude_genres == ("Horror",)
    assert "太压抑" in prefs.unparsed_constraints


def test_agent_returns_grounded_items_and_trace(tools):
    response = DeterministicAgent(tools).run("想看科幻，不要恐怖", user_id=1, top_k=2)
    assert response.mode == "deterministic"
    assert response.latency_ms >= 0
    assert response.status == "ok"
    assert not response.fallback_used
    assert {item.item_id for item in response.items} == {2, 4}
    assert all("Sci-Fi" in item.reason for item in response.items)
    assert all(item.item_id != 1 for item in response.items)
    assert [step.name for step in response.tool_trace] == [
        "get_user_history",
        "recall_candidates",
        "rank_candidates",
    ]
    assert all(step.duration_ms >= 0 and step.status == "ok" for step in response.tool_trace)


def test_agent_empty_match_is_not_failure(tools):
    response = DeterministicAgent(tools).run("想看科幻和恐怖", user_id=None)
    assert response.status == "no_match"
    assert response.items == ()
    assert not response.fallback_used


def test_tool_failure_uses_bounded_fallback(tools, monkeypatch):
    def broken_rank(*args, **kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(tools, "rank_candidates", broken_rank)
    response = DeterministicAgent(tools).run("想看科幻", user_id=1, top_k=2)
    assert response.fallback_used
    assert response.status == "fallback"
    assert any(step.status == "error" for step in response.tool_trace)
    assert all("Horror" not in item.genres for item in response.items)


def test_agent_rejects_fabricated_movie_metadata(tools, monkeypatch):
    def fabricated_rank(*args, **kwargs):
        return [Recommendation(2, "Invented title", ("Sci-Fi",), 1.0, "itemcf")]

    monkeypatch.setattr(tools, "rank_candidates", fabricated_rank)
    response = DeterministicAgent(tools).run("科幻", user_id=1, top_k=1)
    assert response.fallback_used
    assert any(step.status == "error" for step in response.tool_trace)
    assert all(item.title != "Invented title" for item in response.items)


def test_agent_enforces_step_and_result_limits(tools):
    with pytest.raises(ValueError):
        DeterministicAgent(tools, max_steps=0)
    with pytest.raises(ValueError):
        DeterministicAgent(tools).run("科幻", user_id=1, top_k=21)
    response = DeterministicAgent(tools, max_steps=2).run("科幻", user_id=1)
    assert response.fallback_used
    assert len(response.tool_trace) <= 2


def test_agent_cli_returns_json(tmp_path):
    movies = [
        {"item_id": 1, "title": "One", "genres": ["Comedy"]},
        {"item_id": 2, "title": "Two", "genres": ["Sci-Fi"]},
    ]
    (tmp_path / "items.json").write_text(json.dumps(movies), encoding="utf-8")
    with (tmp_path / "interactions_train.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["user_id", "item_id", "timestamp", "rating"])
        writer.writerows([(1, 1, 1, ""), (2, 2, 2, "")])
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_agent_demo.py"
    result = subprocess.run(
        [sys.executable, str(script), "--processed-dir", str(tmp_path), "--query", "科幻"],
        capture_output=True,
        env={**os.environ, "PYTHONIOENCODING": "gbk"},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.decode("utf-8"))
    assert payload["mode"] == "deterministic"
    assert payload["items"][0]["item_id"] == 2
