"""有调用上限、证据检查与回退的确定性推荐流程。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any, TypeVar
from uuid import uuid4

from recagent.agent.policy import Preferences, parse_preferences
from recagent.agent.tools import RecommendationTools
from recagent.engine.baseline import Recommendation

T = TypeVar("T")


class StepLimitError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ToolTrace:
    name: str
    arguments: dict[str, Any]
    status: str
    duration_ms: float
    result_count: int
    error: str | None = None


@dataclass(frozen=True, slots=True)
class AgentItem:
    item_id: int
    title: str
    genres: tuple[str, ...]
    score: float
    source: str
    reason: str


@dataclass(frozen=True, slots=True)
class AgentResponse:
    mode: str
    status: str
    trace_id: str
    items: tuple[AgentItem, ...]
    tool_trace: tuple[ToolTrace, ...]
    fallback_used: bool
    preferences: Preferences
    latency_ms: float


class DeterministicAgent:
    """无 LLM 的可复现工具流程，不宣称理解词典外的语义。"""

    def __init__(self, tools: RecommendationTools, max_steps: int = 4) -> None:
        if not 1 <= max_steps <= 4:
            raise ValueError("max_steps 必须在 1 到 4 之间")
        self.tools = tools
        self.max_steps = max_steps

    def _fallback(
        self, user_id: int | None, top_k: int, prefs: Preferences
    ) -> tuple[AgentItem, ...]:
        catalog = self.tools.catalog
        seen = set(catalog.history.get(user_id, ())) if user_id is not None else set()
        candidates = tuple(
            item_id
            for item_id in catalog.popularity
            if item_id not in seen
            and set(prefs.genres).issubset(catalog.movies[item_id].genres)
            and not set(prefs.exclude_genres).intersection(catalog.movies[item_id].genres)
        )
        ranked = self.tools.recommender.score_candidates_batch(
            user_id, candidates, mode="popularity"
        )
        return tuple(self._item(item) for item in ranked[:top_k])

    @staticmethod
    def _item(item: Recommendation) -> AgentItem:
        return AgentItem(
            item_id=item.item_id,
            title=item.title,
            genres=item.genres,
            score=item.score,
            source=item.source,
            reason=f"《{item.title}》题材：{', '.join(item.genres)}；排序来源：{item.source}。",
        )

    def run(self, query: str, user_id: int | None = None, top_k: int = 5) -> AgentResponse:
        if not query.strip() or len(query) > 500:
            raise ValueError("query 长度必须在 1 到 500 之间")
        if not 1 <= top_k <= 20:
            raise ValueError("top_k 必须在 1 到 20 之间")
        if user_id is not None and user_id < 1:
            raise ValueError("user_id 必须为正整数")
        request_start = perf_counter()
        prefs = parse_preferences(query)
        trace: list[ToolTrace] = []
        trace_id = uuid4().hex

        def call(name: str, arguments: dict[str, Any], action: Callable[[], list[T]]) -> list[T]:
            if len(trace) >= self.max_steps:
                raise StepLimitError("超过最大工具调用次数")
            start = perf_counter()
            try:
                result = action()
            except Exception as exc:
                trace.append(
                    ToolTrace(
                        name, arguments, "error", (perf_counter() - start) * 1000, 0, str(exc)
                    )
                )
                raise
            trace.append(
                ToolTrace(name, arguments, "ok", (perf_counter() - start) * 1000, len(result))
            )
            return result

        try:
            if user_id is not None:
                call(
                    "get_user_history",
                    {"user_id": user_id, "limit": 10},
                    lambda: self.tools.get_user_history(user_id, limit=10),
                )
            recalled = call(
                "recall_candidates",
                {
                    "user_id": user_id,
                    "genres": prefs.genres,
                    "exclude_genres": prefs.exclude_genres,
                    "limit": 20,
                },
                lambda: self.tools.recall_candidates(
                    user_id, prefs.genres, prefs.exclude_genres, 20
                ),
            )
            if not recalled:
                return AgentResponse(
                    "deterministic",
                    "no_match",
                    trace_id,
                    (),
                    tuple(trace),
                    False,
                    prefs,
                    (perf_counter() - request_start) * 1000,
                )
            allowed_ids = frozenset(item.item_id for item in recalled)
            ranked = call(
                "rank_candidates",
                {
                    "user_id": user_id,
                    "candidate_ids": tuple(item.item_id for item in recalled),
                    "top_k": top_k,
                },
                lambda: self.tools.rank_candidates(
                    user_id, tuple(item.item_id for item in recalled), allowed_ids, top_k
                ),
            )
            if (
                len(ranked) > top_k
                or len({item.item_id for item in ranked}) != len(ranked)
                or any(
                    item.item_id not in allowed_ids
                    or item.title != self.tools.catalog.movies[item.item_id].title
                    or item.genres != self.tools.catalog.movies[item.item_id].genres
                    or item.source not in {"popularity", "itemcf"}
                    or not set(prefs.genres).issubset(item.genres)
                    or set(prefs.exclude_genres).intersection(item.genres)
                    for item in ranked
                )
            ):
                raise ValueError("排序结果未通过候选或题材证据检查")
            status = "partial" if prefs.unparsed_constraints else "ok"
            return AgentResponse(
                "deterministic",
                status,
                trace_id,
                tuple(self._item(item) for item in ranked),
                tuple(trace),
                False,
                prefs,
                (perf_counter() - request_start) * 1000,
            )
        except Exception as exc:  # noqa: BLE001 - 仅此边界回退到训练期基线
            if len(trace) < self.max_steps and (not trace or trace[-1].status != "error"):
                trace.append(ToolTrace("runtime_validation", {}, "error", 0.0, 0, str(exc)))
            fallback_items = self._fallback(user_id, top_k, prefs)
            return AgentResponse(
                "deterministic",
                "fallback",
                trace_id,
                fallback_items,
                tuple(trace),
                True,
                prefs,
                (perf_counter() - request_start) * 1000,
            )
