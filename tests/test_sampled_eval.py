"""固定候选评测的公式、跳过原因与题集校验。"""

import csv
import json
import subprocess
import sys
from math import log2
from pathlib import Path

import pytest

from recagent.eval.sampled import evaluate_sampled


@pytest.fixture
def sampled_dir(tmp_path):
    movies = [
        {"item_id": item_id, "title": str(item_id), "genres": ["Comedy"]} for item_id in range(1, 6)
    ]
    (tmp_path / "items.json").write_text(json.dumps(movies), encoding="utf-8")
    train = [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4), (5, 2, 5), (6, 3, 6), (7, 5, 7)]
    test = [(1, 3, 20), (1, 4, 21), (99, 2, 22), (100, 9, 23)]
    for name, rows in (("train", train), ("test", test)):
        with (tmp_path / f"interactions_{name}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["user_id", "item_id", "timestamp", "rating"])
            writer.writerows((*row, "") for row in rows)
    negatives = {"1": [2, 5], "99": [1, 4], "100": [1, 2]}
    (tmp_path / "eval_negatives_test.json").write_text(json.dumps(negatives), encoding="utf-8")
    return tmp_path


def test_sampled_metrics_and_skipped_users(sampled_dir):
    result = evaluate_sampled(sampled_dir, k=2, expected_negatives=2)
    assert result["evaluated_users"] == 2
    assert result["users_with_train_history"] == 1
    assert result["skipped_unseen_positive"] == 1
    assert result["protocol"] == "sampled_1pos_2neg"
    assert result["popularity"]["recall_at_k"] == 1.0
    assert result["popularity"]["ndcg_at_k"] == pytest.approx((1 / log2(3) + 1) / 2)
    assert result["popularity"]["coverage"] == 0.6


def test_sampled_eval_rejects_positive_collision(sampled_dir):
    target = sampled_dir / "eval_negatives_test.json"
    negatives = json.loads(target.read_text(encoding="utf-8"))
    negatives["1"] = [2, 3]
    target.write_text(json.dumps(negatives), encoding="utf-8")
    with pytest.raises(ValueError, match="碰撞"):
        evaluate_sampled(sampled_dir, expected_negatives=2)


def test_sampled_eval_rejects_collision_with_later_test_positive(sampled_dir):
    target = sampled_dir / "eval_negatives_test.json"
    negatives = json.loads(target.read_text(encoding="utf-8"))
    negatives["1"] = [2, 4]
    target.write_text(json.dumps(negatives), encoding="utf-8")
    with pytest.raises(ValueError, match="碰撞"):
        evaluate_sampled(sampled_dir, expected_negatives=2)


def test_sampled_cli_outputs_json(sampled_dir):
    script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_baseline.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--processed-dir",
            str(sampled_dir),
            "--k",
            "2",
            "--expected-negatives",
            "2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["evaluated_users"] == 2
