"""对已冻结的逐条对话结果做配对失败与回退贡献审计。"""

from __future__ import annotations

from collections import Counter
from typing import Any

STANDARD_EXCLUSIONS = frozenset({"no_match", "tool_fault"})


def audit_runs(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """按任务 ID 对齐多个运行，不重新调用模型或重算任务金标准。"""
    if not runs:
        raise ValueError("没有可审计的运行")

    aligned: dict[str, dict[str, dict[str, Any]]] = {}
    task_signature: dict[str, str] | None = None
    case_hash: str | None = None
    for name, run in runs.items():
        rows = run["cases"]
        by_id = {row["id"]: row for row in rows}
        if len(by_id) != len(rows):
            raise ValueError(f"{name} 包含重复任务 ID")
        if not by_id:
            raise ValueError(f"{name} 没有任务")
        signature = {case_id: row["category"] for case_id, row in by_id.items()}
        digest = run["case_set_sha256"]
        if run["protocol"] != "frozen_dialogue_v1" or (
            task_signature is not None and (signature != task_signature or digest != case_hash)
        ):
            raise ValueError("任务集不一致：协议、哈希、ID 或分类不同")
        task_signature = signature
        case_hash = digest
        aligned[name] = by_id

    assert task_signature is not None and case_hash is not None
    standard_ids = sorted(
        case_id
        for case_id, category in task_signature.items()
        if category not in STANDARD_EXCLUSIONS
    )
    if not standard_ids:
        raise ValueError("没有普通任务")

    summaries: dict[str, dict[str, Any]] = {}
    for name, by_id in aligned.items():
        rows = list(by_id.values())
        standard = [by_id[case_id] for case_id in standard_ids]
        direct_ids = [
            row["id"] for row in standard if row["task_success"] and not row["fallback_used"]
        ]
        rescued_ids = [
            row["id"] for row in standard if row["task_success"] and row["fallback_used"]
        ]
        token_usages = [row["model_usage"] for row in rows if row.get("model_usage")]
        summaries[name] = {
            "total_cases": len(rows),
            "standard_cases": len(standard),
            "standard_direct_success": len(direct_ids),
            "standard_final_success": sum(bool(row["task_success"]) for row in standard),
            "standard_fallback_rescue": len(rescued_ids),
            "direct_success_ids": direct_ids,
            "fallback_rescue_ids": rescued_ids,
            "all_final_success": sum(bool(row["task_success"]) for row in rows),
            "all_fallback": sum(bool(row["fallback_used"]) for row in rows),
            "fallback_causes": dict(
                Counter(
                    row.get("fallback_cause") or "unknown" for row in rows if row["fallback_used"]
                )
            ),
            "token_observed_cases": len(token_usages),
            "total_tokens": sum(int(usage.get("total_tokens", 0)) for usage in token_usages),
        }
    return {
        "protocol": "frozen_dialogue_v1_paired_audit",
        "case_set_sha256": case_hash,
        "standard_case_ids": standard_ids,
        "runs": summaries,
    }
