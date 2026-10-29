"""冻结对话任务的加载与独立评分。"""

import json
from dataclasses import replace

import pytest

from recagent.agent.runtime import DeterministicAgent
from recagent.agent.tools import RecommendationTools
from recagent.engine.store import Catalog, MovieInfo
from recagent.eval.dialogue import DialogueCase, load_cases, score_case, summarize


def test_frozen_suite_has_30_unique_valid_cases():
    cases = load_cases("configs/eval/dialogue_cases.json", min_cases=30)
    assert len(cases) >= 30
    assert len({case.id for case in cases}) == len(cases)
    assert {case.category for case in cases} == {
        "preference",
        "exclusion",
        "history",
        "cold_start",
        "no_match",
        "tool_fault",
    }


def test_loader_rejects_invalid_case(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "x",
                    "category": "preference",
                    "query": "科幻",
                    "user_id": 1,
                    "top_k": 3,
                    "expected_genres": ["Sci-Fi"],
                    "excluded_genres": [],
                    "expect_no_match": False,
                    "inject_rank_failure": False,
                },
                {
                    "id": "x",
                    "category": "preference",
                    "query": "科幻",
                    "user_id": 1,
                    "top_k": 3,
                    "expected_genres": ["Sci-Fi"],
                    "excluded_genres": [],
                    "expect_no_match": False,
                    "inject_rank_failure": False,
                },
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="重复"):
        load_cases(path)


def test_scoring_rejects_fabricated_and_seen_items():
    catalog = Catalog(
        movies={
            1: MovieInfo(1, "Seen", ("Sci-Fi",)),
            2: MovieInfo(2, "Valid", ("Sci-Fi",)),
            3: MovieInfo(3, "Wrong", ("Horror",)),
        },
        history={1: (1,)},
        item_users={1: frozenset({1}), 2: frozenset({2}), 3: frozenset({3})},
        popularity={1: 3, 2: 2, 3: 1},
    )
    case = DialogueCase("p1", "preference", "科幻", 1, 2, ("Sci-Fi",), (), False, False)
    response = DeterministicAgent(RecommendationTools(catalog)).run(case.query, 1, 2)
    good = score_case(case, response, catalog)
    assert good["constraint_pass"] and good["grounding_pass"] and good["task_success"]
    forged = replace(response, items=(replace(response.items[0], title="Invented"),))
    bad = score_case(case, forged, catalog)
    assert not bad["grounding_pass"] and not bad["task_success"]
    seen = replace(response, items=(replace(response.items[0], item_id=1, title="Seen"),))
    assert not score_case(case, seen, catalog)["constraint_pass"]


def test_summary_keeps_failures_and_latency_percentiles():
    rows = [
        {
            "id": "a",
            "category": "preference",
            "constraint_pass": True,
            "grounding_pass": True,
            "preference_parse_pass": True,
            "task_success": True,
            "fallback_used": False,
            "tool_calls": 2,
            "latency_ms": 10.0,
            "failure_reasons": [],
        },
        {
            "id": "b",
            "category": "no_match",
            "constraint_pass": False,
            "grounding_pass": True,
            "preference_parse_pass": True,
            "task_success": False,
            "fallback_used": True,
            "tool_calls": 1,
            "latency_ms": 30.0,
            "failure_reasons": ["expected_no_match"],
        },
    ]
    result = summarize(rows)
    assert result["total"] == 2
    assert result["task_success_rate"] == 0.5
    assert result["fallback_rate"] == 0.5
    assert result["p50_latency_ms"] == 20.0
    assert result["p95_latency_ms"] == 29.0
    assert result["by_category"]["no_match"]["task_success_rate"] == 0.0
    assert result["standard_direct_success_rate"] == 1.0
    assert result["fallback_causes"] == {"unknown": 1}
