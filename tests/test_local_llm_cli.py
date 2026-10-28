"""演示命令在本机模型协议上的串联。"""

import csv
import io
import json
from urllib.error import URLError

from scripts.run_local_llm_agent import main


def test_cli_routes_model_tool_calls(tmp_path, monkeypatch, capsys):
    (tmp_path / "items.json").write_text(
        json.dumps(
            [
                {"item_id": 1, "title": "One", "genres": ["Sci-Fi"]},
                {"item_id": 2, "title": "Two", "genres": ["Sci-Fi"]},
                {"item_id": 4, "title": "Four", "genres": ["Sci-Fi"]},
            ]
        ),
        encoding="utf-8",
    )
    with (tmp_path / "interactions_train.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["user_id", "item_id", "timestamp", "rating"])
        writer.writerows([(1, 1, 1, ""), (2, 2, 2, ""), (3, 4, 3, "")])

    def respond(request, timeout):
        assert timeout > 0
        messages = json.loads(request.data)["messages"]
        calls = [message for message in messages if message["role"] == "tool"]
        if not calls:
            message = {
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "recall_candidates",
                            "arguments": '{"genres":["Sci-Fi"]}',
                        },
                    }
                ],
            }
        elif len(calls) == 1:
            recalled = json.loads(calls[0]["content"])
            ids = [item["item_id"] for item in recalled]
            message = {
                "content": None,
                "tool_calls": [
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {
                            "name": "rank_candidates",
                            "arguments": json.dumps({"candidate_ids": ids, "top_k": 2}),
                        },
                    }
                ],
            }
        else:
            ranked = json.loads(calls[1]["content"])
            message = {"content": json.dumps({"item_ids": [item["item_id"] for item in ranked]})}
        return io.BytesIO(json.dumps({"choices": [{"message": message}]}).encode())

    monkeypatch.setattr("urllib.request.urlopen", respond)
    monkeypatch.setenv("RECAGENT_LLM_MODEL", "fake-local")
    code = main(
        ["--processed-dir", str(tmp_path), "--query", "科幻", "--user-id", "1", "--top-k", "2"]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "llm"
    assert payload["status"] == "ok", payload["tool_trace"]
    assert [step["name"] for step in payload["tool_trace"]] == [
        "recall_candidates",
        "rank_candidates",
    ]
    assert {item["item_id"] for item in payload["items"]} == {2, 4}

    def offline(_request, timeout):
        raise URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", offline)
    assert (
        main(
            ["--processed-dir", str(tmp_path), "--query", "科幻", "--user-id", "1", "--top-k", "2"]
        )
        == 0
    )
    fallback = json.loads(capsys.readouterr().out)
    assert fallback["mode"] == "llm"
    assert fallback["status"] == "fallback"
    assert fallback["fallback_used"] is True
    assert any(step["name"] == "llm_validation" for step in fallback["tool_trace"])
