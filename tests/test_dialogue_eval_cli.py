"""对话任务 CLI 的离线端到端入口。"""

import csv
import json
from urllib.error import URLError

from scripts.evaluate_dialogue_agent import main


def test_cli_writes_case_results_and_summary(tmp_path, monkeypatch):
    (tmp_path / "items.json").write_text(
        json.dumps(
            [
                {"item_id": 1, "title": "Seen", "genres": ["Sci-Fi"]},
                {"item_id": 2, "title": "Valid", "genres": ["Sci-Fi"]},
            ]
        ),
        encoding="utf-8",
    )
    with (tmp_path / "interactions_train.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["user_id", "item_id", "timestamp", "rating"])
        writer.writerows([(1, 1, 1, ""), (2, 2, 2, "")])
    cases = [
        {
            "id": "p",
            "category": "preference",
            "query": "科幻",
            "user_id": 1,
            "top_k": 1,
            "expected_genres": ["Sci-Fi"],
            "excluded_genres": [],
            "expect_no_match": False,
            "inject_rank_failure": False,
        },
        {
            "id": "f",
            "category": "tool_fault",
            "query": "科幻",
            "user_id": 1,
            "top_k": 1,
            "expected_genres": ["Sci-Fi"],
            "excluded_genres": [],
            "expect_no_match": False,
            "inject_rank_failure": True,
        },
    ]
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps(cases), encoding="utf-8")
    output = tmp_path / "result.json"
    assert (
        main(
            [
                "--processed-dir",
                str(tmp_path),
                "--cases",
                str(cases_path),
                "--mode",
                "deterministic",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["summary"]["total"] == 2
    assert result["summary"]["task_success_rate"] == 1.0
    assert result["cases"][1]["fallback_used"] is True
    assert result["cases"][1]["tool_trace"][-1]["name"] == "rank_candidates"

    def offline(_request, timeout):
        raise URLError("offline")

    monkeypatch.setattr("urllib.request.urlopen", offline)
    llm_output = tmp_path / "llm.json"
    assert (
        main(
            [
                "--processed-dir",
                str(tmp_path),
                "--cases",
                str(cases_path),
                "--mode",
                "llm",
                "--output",
                str(llm_output),
            ]
        )
        == 0
    )
    llm_result = json.loads(llm_output.read_text(encoding="utf-8"))
    assert llm_result["summary"]["total"] == 2
    assert llm_result["cases"][0]["fallback_cause"] == "model_unavailable"
