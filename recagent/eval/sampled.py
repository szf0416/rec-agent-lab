"""同一固定候选集合上的单正样本排序评测。"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from math import log2
from pathlib import Path
from typing import Any, Literal

from recagent.engine.baseline import BaselineRecommender
from recagent.engine.store import Catalog


def evaluate_sampled(
    processed_dir: Path, k: int = 10, expected_negatives: int = 100
) -> dict[str, Any]:
    """评估采样题集；NDCG/Recall 不代表全库检索质量。"""
    if k < 1 or expected_negatives < 1:
        raise ValueError("k 和 expected_negatives 必须为正整数")
    catalog = Catalog.load(processed_dir)
    recommender = BaselineRecommender(catalog)
    tests: dict[int, list[tuple[int, int]]] = defaultdict(list)
    with (processed_dir / "interactions_test.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            tests[int(row["user_id"])].append((int(row["timestamp"]), int(row["item_id"])))
    negatives_raw = json.loads(
        (processed_dir / "eval_negatives_test.json").read_text(encoding="utf-8")
    )
    negatives = {
        int(user_id): tuple(int(item) for item in items) for user_id, items in negatives_raw.items()
    }

    totals: dict[str, dict[str, Any]] = {
        mode: {"hits": 0, "dcg": 0.0, "recommended": set()} for mode in ("popularity", "itemcf")
    }
    evaluated = 0
    users_with_train_history = 0
    skipped_unseen_positive = 0
    skipped_seen_positive = 0
    for user_id in sorted(tests):
        seen = set(catalog.history.get(user_id, ()))
        ordered = sorted(tests[user_id])
        eligible = [item for _, item in ordered if item in catalog.popularity and item not in seen]
        if not eligible:
            if any(item in catalog.popularity for _, item in ordered):
                skipped_seen_positive += 1
            else:
                skipped_unseen_positive += 1
            continue
        positive = eligible[0]
        users_with_train_history += bool(seen)
        if user_id not in negatives:
            raise ValueError(f"缺少用户 {user_id} 的固定负样本")
        user_negatives = negatives[user_id]
        if len(user_negatives) != expected_negatives or len(set(user_negatives)) != len(
            user_negatives
        ):
            raise ValueError(f"用户 {user_id} 的负样本数量或唯一性不符合协议")
        all_test_positives = {item for _, item in ordered}
        if (seen | all_test_positives).intersection(user_negatives):
            raise ValueError(f"用户 {user_id} 的正负样本碰撞")
        if any(item not in catalog.popularity for item in user_negatives):
            raise ValueError(f"用户 {user_id} 的负样本包含训练期不可见电影")
        candidates = (positive, *user_negatives)
        for mode in ("popularity", "itemcf"):
            typed_mode: Literal["popularity", "itemcf"] = mode
            ranking = recommender.score_candidates_batch(user_id, candidates, mode=typed_mode)
            top = ranking[:k]
            totals[mode]["recommended"].update(item.item_id for item in top)
            for rank, item in enumerate(top, start=1):
                if item.item_id == positive:
                    totals[mode]["hits"] += 1
                    totals[mode]["dcg"] += 1 / log2(rank + 1)
                    break
        evaluated += 1
    if not evaluated:
        raise ValueError("没有符合协议的测试用户")

    result: dict[str, Any] = {
        "protocol": f"sampled_1pos_{expected_negatives}neg",
        "k": k,
        "seed": 42,
        "test_users": len(tests),
        "evaluated_users": evaluated,
        "users_with_train_history": users_with_train_history,
        "skipped_unseen_positive": skipped_unseen_positive,
        "skipped_seen_positive": skipped_seen_positive,
        "candidate_count_per_user": expected_negatives + 1,
        "catalog_items": len(catalog.popularity),
    }
    for mode, values in totals.items():
        result[mode] = {
            "recall_at_k": values["hits"] / evaluated,
            "ndcg_at_k": values["dcg"] / evaluated,
            "coverage": len(values["recommended"]) / len(catalog.popularity),
        }
    return result
