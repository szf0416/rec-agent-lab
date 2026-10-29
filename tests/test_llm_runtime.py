"""模型工具循环必须真正路由工具，并约束输出证据。"""

import json
import time

import pytest

from recagent.agent.chat_client import ChatTurn, ModelUnavailableError, ToolCall
from recagent.agent.llm_runtime import LLMRecommendationAgent
from recagent.agent.tools import RecommendationTools
from recagent.engine.store import Catalog, MovieInfo


@pytest.fixture
def rec_tools():
    catalog = Catalog(
        movies={
            1: MovieInfo(1, "One", ("Sci-Fi",)),
            2: MovieInfo(2, "Two", ("Sci-Fi", "Comedy")),
            3: MovieInfo(3, "Three", ("Horror",)),
            4: MovieInfo(4, "Four", ("Sci-Fi",)),
        },
        history={1: (1,), 2: (2, 4)},
        item_users={1: frozenset({1}), 2: frozenset({2}), 3: frozenset(), 4: frozenset({2})},
        popularity={1: 2, 2: 2, 3: 1, 4: 1},
    )
    return RecommendationTools(catalog)


class ScriptedClient:
    def __init__(self, turns):
        self.turns = list(turns)
        self.messages = []
        self.tools = []

    def complete(self, messages, tools, timeout_s):
        self.messages.append(json.loads(json.dumps(messages)))
        self.tools.append(tools)
        assert timeout_s > 0
        turn = self.turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return turn


def call(name, arguments, call_id="c1"):
    return ChatTurn(None, (ToolCall(call_id, name, json.dumps(arguments)),), {"total_tokens": 5})


def test_llm_routes_recall_and_rank_with_grounded_output(rec_tools):
    client = ScriptedClient(
        [
            call(
                "recall_candidates",
                {"genres": ["Sci-Fi"], "exclude_genres": ["Horror"], "limit": 10},
            ),
            call("rank_candidates", {"candidate_ids": [2, 4], "top_k": 2}, "c2"),
            ChatTurn('{"item_ids":[2,4],"reason":"fabricated"}', (), {"total_tokens": 3}),
        ]
    )
    result = LLMRecommendationAgent(rec_tools, client).run("想看科幻，不要恐怖", user_id=1, top_k=2)
    assert result.mode == "llm"
    assert result.status == "ok"
    assert not result.fallback_used
    assert [item.item_id for item in result.items] == [2, 4]
    assert "fabricated" not in result.items[0].reason
    assert [step.name for step in result.tool_trace] == ["recall_candidates", "rank_candidates"]
    assert result.model_usage["total_tokens"] == 13
    assert [tool["function"]["name"] for tool in client.tools[0]] == ["recall_candidates"]
    assert client.messages[1][-1]["role"] == "tool"
    assert client.messages[1][-1]["tool_call_id"] == "c1"
    assert client.messages[1][-2]["tool_calls"][0]["function"]["name"] == "recall_candidates"
    assert all(
        tool["function"]["name"] != "get_user_history"
        or tool["function"]["parameters"]["properties"].get("user_id") is None
        for tool in client.tools[0]
    )


@pytest.mark.parametrize(
    "bad_turn",
    [
        call("get_user_history", {"user_id": 2, "limit": 10}),
        call("rank_candidates", {"candidate_ids": [3], "top_k": 2}),
        call("unknown_tool", {}),
        ChatTurn(None, (ToolCall("c1", "recall_candidates", "{"),), {}),
        ChatTurn(
            None,
            (ToolCall("c1", "recall_candidates", "{}"), ToolCall("c2", "rank_candidates", "{}")),
            {},
        ),
        ModelUnavailableError("offline"),
    ],
)
def test_invalid_or_unavailable_model_falls_back(rec_tools, bad_turn):
    result = LLMRecommendationAgent(rec_tools, ScriptedClient([bad_turn])).run(
        "科幻", user_id=1, top_k=2
    )
    assert result.mode == "llm"
    assert result.status == "fallback"
    assert result.fallback_used
    assert all(item.item_id != 1 and "Sci-Fi" in item.genres for item in result.items)
    assert any(step.status == "error" for step in result.tool_trace)


def test_fabricated_final_id_falls_back(rec_tools):
    client = ScriptedClient(
        [
            call("recall_candidates", {"genres": ["Sci-Fi"], "exclude_genres": [], "limit": 10}),
            call("rank_candidates", {"candidate_ids": [2, 4], "top_k": 2}, "c2"),
            ChatTurn('{"item_ids":[3]}', (), {}),
        ]
    )
    result = LLMRecommendationAgent(rec_tools, client).run("科幻", user_id=1, top_k=2)
    assert result.fallback_used
    assert all(item.item_id != 3 for item in result.items)


def test_tool_budget_and_turn_budget_fall_back(rec_tools):
    client = ScriptedClient([call("search_movies", {"query": "Sci-Fi", "limit": 1})] * 5)
    result = LLMRecommendationAgent(rec_tools, client).run("科幻", user_id=1)
    assert result.fallback_used
    assert len([step for step in result.tool_trace if step.status == "ok"]) <= 4


def test_tool_phases_require_recall_then_rank(rec_tools):
    client = ScriptedClient(
        [
            call("search_movies", {"query": "Sci-Fi", "limit": 2}),
            call("recall_candidates", {"genres": ["Sci-Fi"], "limit": 10}, "c2"),
            call("rank_candidates", {"candidate_ids": [2, 4], "top_k": 2}, "c3"),
            ChatTurn('{"item_ids":[2,4]}', (), {}),
        ]
    )
    result = LLMRecommendationAgent(rec_tools, client).run("推荐电影", user_id=1, top_k=2)
    assert result.status == "ok"
    tool_names = [[tool["function"]["name"] for tool in offered] for offered in client.tools]
    assert "search_movies" in tool_names[0]
    assert tool_names[1] == ["recall_candidates"]
    assert tool_names[2] == ["rank_candidates"]
    assert tool_names[3] == []


def test_model_text_before_recall_gets_one_repair_prompt(rec_tools):
    client = ScriptedClient(
        [
            ChatTurn("可以看看科幻电影", (), {}),
            call("recall_candidates", {"genres": ["Sci-Fi"], "limit": 10}),
            call("rank_candidates", {"candidate_ids": [2, 4], "top_k": 2}, "c2"),
            ChatTurn('{"item_ids":[2,4]}', (), {}),
        ]
    )
    result = LLMRecommendationAgent(rec_tools, client).run("科幻", user_id=1, top_k=2)
    assert result.status == "ok"
    assert "recall_candidates" in client.messages[1][-1]["content"]


def test_rank_pool_may_exceed_final_top_k(rec_tools):
    client = ScriptedClient(
        [
            call("recall_candidates", {"genres": ["Sci-Fi"], "limit": 10}),
            call("rank_candidates", {"candidate_ids": [2, 4], "top_k": 2}, "c2"),
            ChatTurn('{"item_ids":[2]}', (), {}),
        ]
    )
    result = LLMRecommendationAgent(rec_tools, client).run("科幻", user_id=1, top_k=1)
    assert result.status == "ok"
    assert [item.item_id for item in result.items] == [2]


def test_final_turn_after_total_deadline_falls_back(rec_tools):
    class SlowFinalClient(ScriptedClient):
        def complete(self, messages, tools, timeout_s):
            if len(self.messages) == 2:
                time.sleep(0.03)
            return super().complete(messages, tools, timeout_s)

    client = SlowFinalClient(
        [
            call("recall_candidates", {"genres": ["Sci-Fi"], "limit": 10}),
            call("rank_candidates", {"candidate_ids": [2, 4], "top_k": 2}, "c2"),
            ChatTurn('{"item_ids":[2,4]}', (), {}),
        ]
    )
    result = LLMRecommendationAgent(rec_tools, client, total_timeout_s=0.02).run(
        "科幻", user_id=1, top_k=2
    )
    assert result.fallback_used
