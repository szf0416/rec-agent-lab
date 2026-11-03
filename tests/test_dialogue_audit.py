"""同一冻结任务集的配对失败审计。"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from recagent.eval.audit import audit_runs


def run(rows, digest="abc"):
    return {"protocol": "frozen_dialogue_v1", "case_set_sha256": digest, "cases": rows}


def row(case_id, category, success, fallback, cause=None, tokens=None):
    return {
        "id": case_id,
        "category": category,
        "task_success": success,
        "fallback_used": fallback,
        "fallback_cause": cause,
        "model_usage": {"total_tokens": tokens} if tokens is not None else None,
        "latency_ms": 100.0,
    }


def test_audit_separates_direct_success_and_fallback_rescue():
    baseline = run(
        [
            row("a", "preference", True, False),
            row("b", "history", True, False),
            row("c", "no_match", True, False),
            row("d", "tool_fault", True, True, "injected_rank_failure"),
        ]
    )
    model = run(
        [
            row("a", "preference", True, False, tokens=100),
            row("b", "history", True, True, "model_protocol_after_recall", tokens=200),
            row("c", "no_match", True, True, "model_protocol_before_recall", tokens=30),
            row("d", "tool_fault", True, True, "injected_rank_failure", tokens=20),
        ]
    )

    result = audit_runs({"baseline": baseline, "model": model})

    assert result["case_set_sha256"] == "abc"
    assert result["standard_case_ids"] == ["a", "b"]
    assert result["runs"]["baseline"]["standard_direct_success"] == 2
    assert result["runs"]["model"]["standard_direct_success"] == 1
    assert result["runs"]["model"]["standard_final_success"] == 2
    assert result["runs"]["model"]["standard_fallback_rescue"] == 1
    assert result["runs"]["model"]["fallback_causes"] == {
        "model_protocol_after_recall": 1,
        "model_protocol_before_recall": 1,
        "injected_rank_failure": 1,
    }
    assert result["runs"]["model"]["total_tokens"] == 350


def test_audit_rejects_mismatched_task_set():
    first = run([row("a", "preference", True, False)])
    changed_hash = run([row("a", "preference", True, False)], digest="other")
    changed_id = run([row("b", "preference", True, False)])
    changed_category = run([row("a", "history", True, False)])

    for other in (changed_hash, changed_id, changed_category):
        with pytest.raises(ValueError, match="任务集不一致"):
            audit_runs({"first": first, "other": other})


def test_audit_rejects_duplicate_ids_and_empty_runs():
    with pytest.raises(ValueError):
        audit_runs({})
    duplicate = run([row("a", "preference", True, False)] * 2)
    with pytest.raises(ValueError, match="重复"):
        audit_runs({"duplicate": duplicate})


def test_cli_writes_reproducible_json(tmp_path):
    fixture = run([row("a", "preference", True, False)])
    source = tmp_path / "run.json"
    output = tmp_path / "audit.json"
    source.write_text(json.dumps(fixture), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_dialogue_runs.py"

    result = subprocess.run(
        [sys.executable, str(script), "--baseline", str(source), "--output", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (
        json.loads(output.read_text(encoding="utf-8"))["runs"]["baseline"][
            "standard_direct_success"
        ]
        == 1
    )
