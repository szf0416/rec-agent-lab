"""冻结对话推荐任务和独立评分。"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from recagent.agent.policy import GENRE_ALIASES
from recagent.agent.runtime import AgentResponse
from recagent.engine.store import Catalog

CATEGORIES = frozenset(
    {"preference", "exclusion", "history", "cold_start", "no_match", "tool_fault"}
)
CASE_FIELDS = frozenset(
    {
        "id",
        "category",
        "query",
        "user_id",
        "top_k",
        "expected_genres",
        "excluded_genres",
        "expect_no_match",
        "inject_rank_failure",
    }
)


@dataclass(frozen=True, slots=True)
class DialogueCase:
    id: str
    category: str
    query: str
    user_id: int | None
    top_k: int
    expected_genres: tuple[str, ...]
    excluded_genres: tuple[str, ...]
    expect_no_match: bool
    inject_rank_failure: bool


def load_cases(path: str | Path, min_cases: int = 1) -> tuple[DialogueCase, ...]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) < min_cases:
        raise ValueError("冻结任务数量不足")
    cases: list[DialogueCase] = []
    seen: set[str] = set()
    valid_genres = frozenset(GENRE_ALIASES.values())
    for row in raw:
        if not isinstance(row, dict) or set(row) != CASE_FIELDS:
            raise ValueError("任务字段不符合协议")
        if not isinstance(row["id"], str) or not row["id"]:
            raise ValueError("任务 ID 无效")
        if row["id"] in seen:
            raise ValueError("任务 ID 重复")
        seen.add(row["id"])
        if (
            row["category"] not in CATEGORIES
            or not isinstance(row["query"], str)
            or not 1 <= len(row["query"]) <= 500
        ):
            raise ValueError("任务分类或查询无效")
        user_id = row["user_id"]
        if user_id is not None and (type(user_id) is not int or user_id < 1):
            raise ValueError("用户 ID 无效")
        if type(row["top_k"]) is not int or not 1 <= row["top_k"] <= 20:
            raise ValueError("top_k 无效")
        for field in ("expected_genres", "excluded_genres"):
            value: Any = row[field]
            if (
                not isinstance(value, list)
                or len(value) > 5
                or len(set(value)) != len(value)
                or any(not isinstance(item, str) or item not in valid_genres for item in value)
            ):
                raise ValueError("题材金标准无效")
        if type(row["expect_no_match"]) is not bool or type(row["inject_rank_failure"]) is not bool:
            raise ValueError("任务预期标记无效")
        if row["inject_rank_failure"] != (row["category"] == "tool_fault"):
            raise ValueError("故障标记与分类不一致")
        if row["expect_no_match"] != (row["category"] == "no_match"):
            raise ValueError("无匹配标记与分类不一致")
        cases.append(
            DialogueCase(
                id=row["id"],
                category=row["category"],
                query=row["query"],
                user_id=user_id,
                top_k=row["top_k"],
                expected_genres=tuple(row["expected_genres"]),
                excluded_genres=tuple(row["excluded_genres"]),
                expect_no_match=row["expect_no_match"],
                inject_rank_failure=row["inject_rank_failure"],
            )
        )
    return tuple(cases)


def score_case(case: DialogueCase, response: AgentResponse, catalog: Catalog) -> dict[str, Any]:
    """只用冻结金标准和训练期目录检查结果，不从查询重算答案。"""
    seen = set(catalog.history.get(case.user_id, ())) if case.user_id is not None else set()
    ids = [item.item_id for item in response.items]
    constraints = (
        len(ids) <= case.top_k
        and len(ids) == len(set(ids))
        and (not ids if case.expect_no_match else bool(ids))
        and all(
            item.item_id not in seen
            and set(case.expected_genres).issubset(item.genres)
            and not set(case.excluded_genres).intersection(item.genres)
            for item in response.items
        )
    )
    grounding = all(
        item.item_id in catalog.movies
        and item.title == catalog.movies[item.item_id].title
        and item.genres == catalog.movies[item.item_id].genres
        and item.source in {"popularity", "itemcf"}
        and item.title in item.reason
        and item.source in item.reason
        and all(genre in item.reason for genre in item.genres)
        for item in response.items
    )
    preference_parse = set(response.preferences.genres) == set(case.expected_genres) and set(
        response.preferences.exclude_genres
    ) == set(case.excluded_genres)
    fallback_expected = not case.inject_rank_failure or (
        response.fallback_used
        and any(
            step.name == "rank_candidates" and step.status == "error"
            for step in response.tool_trace
        )
    )
    success = constraints and grounding and preference_parse and fallback_expected
    failures = [
        name
        for name, passed in (
            ("constraint", constraints),
            ("grounding", grounding),
            ("preference_parse", preference_parse),
            ("expected_fallback", fallback_expected),
        )
        if not passed
    ]
    fallback_cause: str | None = None
    if response.fallback_used:
        if case.inject_rank_failure:
            fallback_cause = "injected_rank_failure"
        else:
            validation = next(
                (step for step in response.tool_trace if step.name == "llm_validation"), None
            )
            if validation is None:
                fallback_cause = "tool_or_runtime_failure"
            elif validation.error == "ModelUnavailableError":
                fallback_cause = "model_unavailable"
            elif validation.error == "ModelDecisionError":
                prior = response.tool_trace[: response.tool_trace.index(validation)]
                if any(step.name == "rank_candidates" for step in prior):
                    fallback_cause = "model_protocol_after_rank"
                elif any(step.name == "recall_candidates" for step in prior):
                    fallback_cause = "model_protocol_after_recall"
                else:
                    fallback_cause = "model_protocol_before_recall"
            else:
                fallback_cause = "model_or_tool_error"
    return {
        "id": case.id,
        "category": case.category,
        "query": case.query,
        "user_id": case.user_id,
        "mode": response.mode,
        "status": response.status,
        "item_ids": ids,
        "constraint_pass": constraints,
        "grounding_pass": grounding,
        "preference_parse_pass": preference_parse,
        "task_success": success,
        "fallback_used": response.fallback_used,
        "fallback_cause": fallback_cause,
        "tool_calls": sum(step.status == "ok" for step in response.tool_trace),
        "latency_ms": response.latency_ms,
        "model_usage": response.model_usage,
        "tool_trace": [asdict(step) for step in response.tool_trace],
        "failure_reasons": failures,
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("没有评测结果")

    def rate(group: list[dict[str, Any]], field: str) -> float:
        return sum(bool(row[field]) for row in group) / len(group)

    by_category: dict[str, dict[str, Any]] = {}
    standard = [row for row in rows if row["category"] not in {"no_match", "tool_fault"}]
    for category in sorted({row["category"] for row in rows}):
        group = [row for row in rows if row["category"] == category]
        by_category[category] = {
            "total": len(group),
            "task_success_rate": rate(group, "task_success"),
            "fallback_rate": rate(group, "fallback_used"),
        }
    return {
        "total": len(rows),
        "constraint_pass_rate": rate(rows, "constraint_pass"),
        "grounding_pass_rate": rate(rows, "grounding_pass"),
        "preference_parse_pass_rate": rate(rows, "preference_parse_pass"),
        "task_success_rate": rate(rows, "task_success"),
        "standard_direct_success_rate": (
            sum(bool(row["task_success"]) and not row["fallback_used"] for row in standard)
            / len(standard)
            if standard
            else 0.0
        ),
        "fallback_rate": rate(rows, "fallback_used"),
        "fallback_causes": dict(
            Counter(row.get("fallback_cause") or "unknown" for row in rows if row["fallback_used"])
        ),
        "mean_tool_calls": sum(row["tool_calls"] for row in rows) / len(rows),
        "p50_latency_ms": _percentile([row["latency_ms"] for row in rows], 0.5),
        "p95_latency_ms": _percentile([row["latency_ms"] for row in rows], 0.95),
        "by_category": by_category,
        "failed_ids": [row["id"] for row in rows if not row["task_success"]],
    }
