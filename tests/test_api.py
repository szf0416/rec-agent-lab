"""本地 HTTP 演示入口的请求契约与模型回退。"""

import csv
import json

import pytest
from fastapi.testclient import TestClient

from recagent.agent.chat_client import ModelUnavailableError
from recagent.serving.api import create_app


@pytest.fixture
def processed_dir(tmp_path):
    movies = [
        {"item_id": 1, "title": "Seen", "genres": ["Sci-Fi"]},
        {"item_id": 2, "title": "Space", "genres": ["Sci-Fi"]},
        {"item_id": 3, "title": "Scary", "genres": ["Horror"]},
    ]
    (tmp_path / "items.json").write_text(json.dumps(movies), encoding="utf-8")
    with (tmp_path / "interactions_train.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["user_id", "item_id", "timestamp", "rating"])
        writer.writerows([(1, 1, 1, 5), (2, 2, 2, 5), (3, 2, 3, 4), (4, 3, 4, 5)])
    return tmp_path


def test_health_reports_loaded_catalog(processed_dir):
    client = TestClient(create_app(processed_dir))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "catalog_items": 3}


def test_baseline_recommendation_preserves_agent_trace(processed_dir):
    client = TestClient(create_app(processed_dir))

    response = client.post(
        "/recommend",
        json={"query": "想看科幻，不要恐怖", "user_id": 1, "top_k": 2, "mode": "baseline"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "deterministic"
    assert body["status"] == "ok"
    assert body["fallback_used"] is False
    assert len(body["trace_id"]) == 32
    assert body["latency_ms"] >= 0
    assert [item["item_id"] for item in body["items"]] == [2]
    assert {step["name"] for step in body["tool_trace"]} == {
        "get_user_history",
        "recall_candidates",
        "rank_candidates",
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "   "},
        {"query": "x" * 501},
        {"query": "科幻", "top_k": 0},
        {"query": "科幻", "top_k": 21},
        {"query": "科幻", "user_id": 0},
        {"query": "科幻", "mode": "other"},
        {"query": "科幻", "processed_dir": "C:/private"},
    ],
)
def test_invalid_request_is_structured_422(processed_dir, payload):
    response = TestClient(create_app(processed_dir)).post("/recommend", json=payload)

    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)


class UnavailableClient:
    def complete(self, messages, tools, timeout_s):
        raise ModelUnavailableError("offline")


def test_llm_unavailable_returns_traced_fallback(processed_dir):
    response = TestClient(create_app(processed_dir, client=UnavailableClient())).post(
        "/recommend", json={"query": "想看科幻", "user_id": 1, "mode": "llm"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "llm"
    assert body["fallback_used"] is True
    assert body["status"] == "fallback"
    assert body["items"][0]["item_id"] == 2
    assert any(step["status"] == "error" for step in body["tool_trace"])


def test_missing_catalog_fails_at_app_creation(tmp_path):
    with pytest.raises(FileNotFoundError):
        create_app(tmp_path)


def test_internal_error_is_structured_without_configuration_leak(processed_dir, monkeypatch):
    monkeypatch.setenv("RECAGENT_LLM_BASE_URL", "https://example.org/private-token")
    client = TestClient(create_app(processed_dir), raise_server_exceptions=False)

    response = client.post("/recommend", json={"query": "科幻", "mode": "llm"})

    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error", "message": "推荐服务处理失败"}}
    assert "private-token" not in response.text
