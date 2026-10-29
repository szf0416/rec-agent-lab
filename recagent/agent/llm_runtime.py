"""有预算和证据约束的本地模型推荐工具循环。"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from time import perf_counter
from typing import Any
from uuid import uuid4

from recagent.agent.chat_client import ChatClient, ChatTurn, ToolCall
from recagent.agent.policy import parse_preferences
from recagent.agent.runtime import AgentResponse, DeterministicAgent, ToolTrace
from recagent.agent.tools import RecommendationTools
from recagent.engine.baseline import Recommendation


class ModelDecisionError(RuntimeError):
    """模型给出不符合工具或结果协议的决定。"""


def _schema(
    name: str, description: str, properties: dict[str, Any], required: list[str]
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


TOOL_SCHEMAS = [
    _schema(
        "search_movies",
        "按片名或题材搜索目录",
        {"query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 20}},
        ["query"],
    ),
    _schema(
        "get_user_history",
        "读取当前用户的历史电影；不能指定用户 ID",
        {"limit": {"type": "integer", "minimum": 1, "maximum": 20}},
        [],
    ),
    _schema(
        "recall_candidates",
        "根据题材和当前用户召回候选",
        {
            "genres": {"type": "array", "items": {"type": "string"}},
            "exclude_genres": {"type": "array", "items": {"type": "string"}},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        [],
    ),
    _schema(
        "rank_candidates",
        "只对上一次召回的候选 ID 排序",
        {
            "candidate_ids": {"type": "array", "items": {"type": "integer"}},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        ["candidate_ids"],
    ),
]


def _bounded_int(value: Any, upper: int) -> int:
    if type(value) is not int or not 1 <= value <= upper:
        raise ModelDecisionError("工具数量参数无效")
    return value


def _genres(value: Any, allowed: frozenset[str]) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or len(value) > 5
        or any(not isinstance(item, str) or item not in allowed for item in value)
    ):
        raise ModelDecisionError("题材参数无效")
    return tuple(sorted(set(value)))


def _normalize_text_tool_call(turn: ChatTurn, offered: list[dict[str, Any]]) -> ChatTurn:
    """兼容将函数调用写在正文 JSON 中的本地模型，仍由运行时校验权限。"""
    if turn.tool_calls or not turn.content or not offered:
        return turn
    try:
        envelope = json.loads(turn.content)
    except ValueError:
        return turn
    if not isinstance(envelope, dict) or set(envelope) != {"name", "arguments"}:
        return turn
    name = envelope["name"]
    arguments = envelope["arguments"]
    if not isinstance(name, str) or name not in {schema["function"]["name"] for schema in offered}:
        raise ModelDecisionError("正文工具调用未获当前阶段授权")
    if not isinstance(arguments, dict):
        raise ModelDecisionError("正文工具参数必须是对象")
    call = ToolCall("text_" + uuid4().hex, name, json.dumps(arguments, ensure_ascii=False))
    return ChatTurn(None, (call,), turn.usage)


class LLMRecommendationAgent:
    def __init__(
        self,
        tools: RecommendationTools,
        client: ChatClient,
        max_tool_calls: int = 4,
        max_turns: int = 6,
        total_timeout_s: float = 60,
    ) -> None:
        if not 1 <= max_tool_calls <= 4 or not 1 <= max_turns <= 6 or total_timeout_s <= 0:
            raise ValueError("模型调用预算无效")
        self.tools = tools
        self.client = client
        self.max_tool_calls = max_tool_calls
        self.max_turns = max_turns
        self.total_timeout_s = total_timeout_s

    def _execute(
        self,
        call: ToolCall,
        user_id: int | None,
        top_k: int,
        required_genres: tuple[str, ...],
        excluded_genres: tuple[str, ...],
        allowed_ids: frozenset[int],
    ) -> tuple[list[Any], dict[str, Any]]:
        try:
            args = json.loads(call.arguments)
        except ValueError:
            raise ModelDecisionError("工具参数不是 JSON") from None
        if not isinstance(args, dict):
            raise ModelDecisionError("工具参数必须是对象")
        keys = {
            "search_movies": {"query", "limit"},
            "get_user_history": {"limit"},
            "recall_candidates": {"genres", "exclude_genres", "limit"},
            "rank_candidates": {"candidate_ids", "top_k"},
        }
        if call.name not in keys or not set(args).issubset(keys[call.name]):
            raise ModelDecisionError("未知工具或额外参数")
        if call.name == "search_movies":
            query = args.get("query")
            if not isinstance(query, str):
                raise ModelDecisionError("搜索词无效")
            limit = _bounded_int(args.get("limit", 10), 20)
            return self.tools.search_movies(query, limit), {"query": query, "limit": limit}
        if call.name == "get_user_history":
            if user_id is None:
                raise ModelDecisionError("当前请求无用户 ID")
            limit = _bounded_int(args.get("limit", 10), 20)
            return self.tools.get_user_history(user_id, limit), {"user_id": user_id, "limit": limit}
        if call.name == "recall_candidates":
            genre_vocab = frozenset(
                genre for movie in self.tools.catalog.movies.values() for genre in movie.genres
            )
            genres = _genres(args.get("genres", []), genre_vocab)
            excludes = _genres(args.get("exclude_genres", []), genre_vocab)
            genres = tuple(sorted(set(genres) | set(required_genres)))
            excludes = tuple(sorted(set(excludes) | set(excluded_genres)))
            limit = _bounded_int(args.get("limit", 20), 20)
            return self.tools.recall_candidates(user_id, genres, excludes, limit), {
                "user_id": user_id,
                "genres": genres,
                "exclude_genres": excludes,
                "limit": limit,
            }
        if not allowed_ids:
            raise ModelDecisionError("排序前必须召回候选")
        raw_ids = args.get("candidate_ids")
        if (
            not isinstance(raw_ids, list)
            or not raw_ids
            or len(raw_ids) > 20
            or any(type(item_id) is not int for item_id in raw_ids)
            or len(set(raw_ids)) != len(raw_ids)
        ):
            raise ModelDecisionError("排序候选 ID 无效")
        candidate_ids = tuple(raw_ids)
        if not set(candidate_ids).issubset(allowed_ids):
            raise ModelDecisionError("排序候选超出召回范围")
        limit = _bounded_int(args.get("top_k", top_k), 20)
        return self.tools.rank_candidates(user_id, candidate_ids, allowed_ids, limit), {
            "user_id": user_id,
            "candidate_ids": candidate_ids,
            "top_k": limit,
        }

    def run(self, query: str, user_id: int | None = None, top_k: int = 5) -> AgentResponse:
        if (
            not query.strip()
            or len(query) > 500
            or not 1 <= top_k <= 20
            or (user_id is not None and (type(user_id) is not int or user_id < 1))
        ):
            raise ValueError("请求参数无效")
        start = perf_counter()
        prefs = parse_preferences(query)
        trace: list[ToolTrace] = []
        usage: dict[str, int] = {}
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "你是电影推荐工具调度器。先调用 recall_candidates，再调用 rank_candidates。"
                    '只推荐工具返回的电影。最后只输出 JSON 对象：{"item_ids":[整数ID]}。'
                    "用户 ID 由系统固定；不要在工具参数中传 user_id。"
                    f"已解析的必选题材：{list(prefs.genres)}；排除题材：{list(prefs.exclude_genres)}。"
                ),
            },
            {"role": "user", "content": query},
        ]
        allowed_ids: frozenset[int] = frozenset()
        ranked: dict[int, Recommendation] = {}
        seen_call_ids: set[str] = set()
        repair_count = 0
        try:
            for _ in range(self.max_turns):
                remaining = self.total_timeout_s - (perf_counter() - start)
                if remaining <= 0:
                    raise ModelDecisionError("模型总时间预算耗尽")
                if ranked:
                    offered = []
                elif allowed_ids:
                    offered = [
                        schema
                        for schema in TOOL_SCHEMAS
                        if schema["function"]["name"] == "rank_candidates"
                    ]
                elif trace or prefs.genres or prefs.exclude_genres:
                    offered = [
                        schema
                        for schema in TOOL_SCHEMAS
                        if schema["function"]["name"] == "recall_candidates"
                    ]
                else:
                    initial_names = {"search_movies", "recall_candidates"}
                    if user_id is not None:
                        initial_names.add("get_user_history")
                    offered = [
                        schema
                        for schema in TOOL_SCHEMAS
                        if schema["function"]["name"] in initial_names
                    ]
                turn: ChatTurn = self.client.complete(messages, offered, remaining)
                if perf_counter() - start > self.total_timeout_s:
                    raise ModelDecisionError("模型总时间预算耗尽")
                turn = _normalize_text_tool_call(turn, offered)
                for key, value in turn.usage.items():
                    if type(value) is int and value >= 0:
                        usage[key] = usage.get(key, 0) + value
                if turn.tool_calls:
                    if len(turn.tool_calls) != 1 or len(trace) >= self.max_tool_calls:
                        raise ModelDecisionError("模型工具调用次数或并行调用超限")
                    call = turn.tool_calls[0]
                    if call.name not in {schema["function"]["name"] for schema in offered}:
                        raise ModelDecisionError("模型调用了当前阶段未授权的工具")
                    if call.id in seen_call_ids:
                        raise ModelDecisionError("重复工具调用 ID")
                    seen_call_ids.add(call.id)
                    tool_start = perf_counter()
                    result, arguments = self._execute(
                        call, user_id, top_k, prefs.genres, prefs.exclude_genres, allowed_ids
                    )
                    if call.name == "recall_candidates":
                        if not result:
                            raise ModelDecisionError("召回结果为空")
                        allowed_ids = frozenset(item.item_id for item in result)
                        ranked = {}
                    if call.name == "rank_candidates":
                        ranked = {item.item_id: item for item in result}
                        if any(
                            item.item_id not in allowed_ids
                            or item.title != self.tools.catalog.movies[item.item_id].title
                            or item.genres != self.tools.catalog.movies[item.item_id].genres
                            or item.source != "itemcf"
                            or not set(prefs.genres).issubset(item.genres)
                            or set(prefs.exclude_genres).intersection(item.genres)
                            for item in result
                        ):
                            raise ModelDecisionError("排序结果未通过证据检查")
                    trace.append(
                        ToolTrace(
                            call.name,
                            arguments,
                            "ok",
                            (perf_counter() - tool_start) * 1000,
                            len(result),
                        )
                    )
                    messages.append(
                        {
                            "role": "assistant",
                            "content": turn.content,
                            "tool_calls": [
                                {
                                    "id": call.id,
                                    "type": "function",
                                    "function": {"name": call.name, "arguments": call.arguments},
                                }
                            ],
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps(
                                [asdict(item) for item in result], ensure_ascii=False
                            ),
                        }
                    )
                    if call.name == "rank_candidates":
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f'只输出 JSON {{"item_ids":[...]}}，最多 {top_k} 个；'
                                    f"ID 只能从已排序结果 {list(ranked)} 中选。"
                                ),
                            }
                        )
                    continue
                if not ranked and repair_count < 1:
                    repair_count += 1
                    required_tool = "rank_candidates" if allowed_ids else "recall_candidates"
                    messages.append({"role": "assistant", "content": turn.content or ""})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"当前步骤必须调用 {required_tool} 工具。不要直接回答或给出电影 ID。"
                            ),
                        }
                    )
                    continue
                if not ranked or not turn.content:
                    raise ModelDecisionError("模型未完成有效排序")
                try:
                    final = json.loads(turn.content)
                except ValueError:
                    raise ModelDecisionError("最终结果不是 JSON") from None
                ids = final.get("item_ids") if isinstance(final, dict) else None
                if (
                    not isinstance(ids, list)
                    or len(ids) > top_k
                    or not ids
                    or any(type(item_id) is not int or item_id not in ranked for item_id in ids)
                    or len(set(ids)) != len(ids)
                ):
                    raise ModelDecisionError("最终电影 ID 未通过证据检查")
                items = tuple(DeterministicAgent._item(ranked[item_id]) for item_id in ids)
                return AgentResponse(
                    mode="llm",
                    status="partial" if prefs.unparsed_constraints else "ok",
                    trace_id=uuid4().hex,
                    items=items,
                    tool_trace=tuple(trace),
                    fallback_used=False,
                    preferences=prefs,
                    latency_ms=(perf_counter() - start) * 1000,
                    model_usage=usage,
                )
            raise ModelDecisionError("模型请求轮数超限")
        except Exception as exc:  # noqa: BLE001 - 本机模型和工具调用的统一回退边界
            # trace 只保留错误类别，避免将模型原始文本或服务密钥写入输出。
            trace.append(ToolTrace("llm_validation", {}, "error", 0.0, 0, type(exc).__name__))
            fallback = DeterministicAgent(self.tools).run(query, user_id, top_k)
            return replace(
                fallback,
                mode="llm",
                status="fallback",
                fallback_used=True,
                tool_trace=tuple(trace) + fallback.tool_trace,
                latency_ms=(perf_counter() - start) * 1000,
                model_usage=usage,
            )
