"""数据预处理：解析 -> 时间切分 -> 泄漏检查 -> 负采样 -> 导出。

用法:
    python scripts/preprocess.py --config configs/data/ml1m.yaml
    python scripts/preprocess.py --config configs/data/ml1m.yaml --strategy leave_one_out

产出（写入 processed_dir）:
    interactions_train.csv / _valid.csv / _test.csv   切分后的交互
    sequences_train.npz                                训练集用户行为序列（SASRec 用）
    eval_negatives_test.json                           固定评测负样本（全模型共用）
    items.json / users.json                            物品与用户元数据
    summary.json                                       本次运行的统计与泄漏检查结果

设计说明:
- 原始目录只读：本脚本拒绝把输出写进 raw_dir，避免污染可被校验的原始数据。
- 训练负样本不落盘：每组 4 个、百万级交互会生成数百 MB 文本，
  改由训练时用同一 seed 现场采样（可复现），省空间也更快。
- 评测负样本必须落盘：它们是"全模型共用的题集"，必须冻结才能让指标可比。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from recagent.data import (
    IntegrityError,
    NegativeSampler,
    build_candidate_pool,
    build_sequences,
    build_splitter,
    from_ratings,
    item_frequencies,
    leakage_report,
    load_ml1m_cached,
    sequence_stats,
    to_dataframe,
)
from recagent.data.ml1m import export_metadata
from recagent.utils.logging import get_logger
from recagent.utils.seeding import seed_everything

logger = get_logger("preprocess")


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    # 配置带 @package data 指令，顶层即数据字段
    return raw


def deep_get(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


# --------------------------------------------------------------------------
# 导出
# --------------------------------------------------------------------------


def assert_output_separate(raw_dir: Path, processed_dir: Path) -> None:
    """拒绝把处理产物写进原始数据目录（原始数据必须保持只读且可校验）。"""
    raw_resolved = raw_dir.resolve()
    proc_resolved = processed_dir.resolve()
    if proc_resolved == raw_resolved or raw_resolved in proc_resolved.parents:
        raise IntegrityError(
            f"processed_dir ({proc_resolved}) 位于 raw_dir ({raw_resolved}) 之内。\n"
            f"处理产物会污染可校验的原始数据，请把两者分开。"
        )


def export_interactions(split: Any, processed_dir: Path) -> dict[str, str]:
    paths: dict[str, str] = {}
    for name in ("train", "valid", "test"):
        interactions = getattr(split, name)
        frame = to_dataframe(interactions)
        target = processed_dir / f"interactions_{name}.csv"
        frame.to_csv(target, index=False)
        paths[name] = str(target)
        logger.info("导出 %s -> %s (%s 行)", name, target.name, len(frame))
    return paths


def export_sequences(sequences: dict[int, list[int]], processed_dir: Path) -> Path:
    """用 npz 存不规则序列：user_ids 与 items 两个并行数组。"""
    user_ids = np.fromiter(sequences.keys(), dtype=np.int64, count=len(sequences))
    items = np.concatenate([np.asarray(v, dtype=np.int64) for v in sequences.values()])
    lengths = np.fromiter((len(v) for v in sequences.values()), dtype=np.int64)

    target = processed_dir / "sequences_train.npz"
    np.savez_compressed(target, user_ids=user_ids, items=items, lengths=lengths)
    logger.info("导出序列 -> %s (%s 用户, %s 个物品标记)", target.name, len(user_ids), len(items))
    return target


def export_eval_negatives(negatives: dict[int, tuple[int, ...]], processed_dir: Path) -> Path:
    target = processed_dir / "eval_negatives_test.json"
    payload = {str(user): list(items) for user, items in sorted(negatives.items())}
    target.write_text(json.dumps(payload), encoding="utf-8")
    logger.info("导出评测负样本 -> %s (%s 用户)", target.name, len(payload))
    return target


def export_summary(summary: dict[str, Any], processed_dir: Path) -> Path:
    target = processed_dir / "summary.json"
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("导出统计 -> %s", target.name)
    return target


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ML-1M 预处理管线")
    parser.add_argument("--config", default="configs/data/ml1m.yaml", help="数据配置 yaml")
    parser.add_argument("--raw-dir", default=None, help="覆盖原始数据目录")
    parser.add_argument("--processed-dir", default=None, help="覆盖输出目录")
    parser.add_argument(
        "--strategy",
        default=None,
        choices=["temporal", "leave_one_out"],
        help="覆盖切分策略",
    )
    parser.add_argument("--seed", type=int, default=None, help="覆盖随机种子")
    parser.add_argument("--dry-run", action="store_true", help="只跑统计与泄漏检查，不写任何文件")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config = load_config(Path(args.config))

    raw_dir = Path(args.raw_dir or deep_get(config, "raw_dir", default="data/raw/ml-1m"))
    processed_dir = Path(
        args.processed_dir or deep_get(config, "processed_dir", default="data/processed/ml1m")
    )
    strategy = args.strategy or deep_get(config, "split", "strategy", default="temporal")
    seed = args.seed if args.seed is not None else int(deep_get(config, "seed", default=42))

    seed_everything(seed)
    logger.info(
        "配置: raw=%s processed=%s strategy=%s seed=%s", raw_dir, processed_dir, strategy, seed
    )

    try:
        assert_output_separate(raw_dir, processed_dir)
    except IntegrityError as exc:
        logger.error("%s", exc)
        return 2

    # --- 1. 加载（带缓存；dry-run 时不产生任何写入副作用） ---
    try:
        dataset = load_ml1m_cached(raw_dir, processed_dir, use_cache=not args.dry_run)
    except IntegrityError as exc:
        logger.error("加载原始数据失败: %s", exc)
        logger.error("提示: 先运行 python scripts/download_data.py --dataset ml-1m")
        return 1

    logger.info("原始数据统计: %s", dataset.summary())

    # --- 2. 转交互记录 ---
    interactions = from_ratings(dataset.ratings)
    logger.info("交互记录 %s 条", len(interactions))

    # --- 3. 切分 + 泄漏检查 ---
    splitter = build_splitter(
        strategy,
        train_ratio=float(deep_get(config, "split", "train_ratio", default=0.8)),
        valid_ratio=float(deep_get(config, "split", "valid_ratio", default=0.1)),
        min_interactions=int(deep_get(config, "filter", "min_user_interactions", default=3)),
    )
    try:
        split = splitter.split(interactions)
    except Exception as exc:  # noqa: BLE001 - 入口需要可读报告
        logger.error("切分失败: %s", exc)
        return 1

    report = leakage_report(split)
    logger.info("切分结果: %s", report["sizes"])
    logger.info(
        "时间范围: train=%s valid=%s test=%s",
        report["train_range"],
        report["valid_range"],
        report["test_range"],
    )
    if not report["strictly_ordered"]:
        logger.error("切分结果时间上不严格有序，存在泄漏风险，中止")
        return 1
    logger.info(
        "训练期未出现的物品数: %s（这些物品不参与负采样候选池）",
        report["items_unseen_in_train"],
    )

    # --- 4. 序列构造（仅训练集，禁止跨切分） ---
    max_len = int(deep_get(config, "sequence", "max_len", default=50))
    min_len = int(deep_get(config, "sequence", "min_len", default=5))
    sequences = build_sequences(split.train, max_len=max_len, min_len=min_len)
    stats = sequence_stats(sequences)
    logger.info(
        "训练序列: %s 用户, 平均长度 %.1f, 最长 %s",
        int(stats["num_users"]),
        stats["mean_len"],
        int(stats["max_len"]),
    )

    # --- 5. 负采样 ---
    # 约束一：候选池只能包含"训练期已存在"的物品，否则等于用未来物品做负样本
    train_max_ts = max(i.timestamp for i in split.train)
    pool = build_candidate_pool(split.train, cutoff=train_max_ts)
    logger.info("负采样候选池: %s 个物品（cutoff=%s）", len(pool), train_max_ts)

    sampler = NegativeSampler(
        pool,
        seed=seed,
        strategy=str(deep_get(config, "negative_sampling", "strategy", default="uniform")),
        frequencies=item_frequencies(split.train),
    )
    eval_num_neg = int(deep_get(config, "negative_sampling", "eval_num_neg", default=100))

    test_users = sorted({i.user_id for i in split.test})
    try:
        eval_negatives = sampler.eval_negatives(test_users, num_neg=eval_num_neg)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "评测负采样失败: %s\n提示: 池大小 %s 可能小于 eval_num_neg=%s",
            exc,
            len(pool),
            eval_num_neg,
        )
        return 1
    logger.info(
        "评测负样本: %s 用户 × %s 个（已冻结，供所有模型共用）", len(eval_negatives), eval_num_neg
    )

    # --- 6. 汇总 ---
    summary: dict[str, Any] = {
        "dataset": deep_get(config, "name", default="ml1m"),
        "strategy": strategy,
        "seed": seed,
        "raw_dir": str(raw_dir),
        "processed_dir": str(processed_dir),
        "dataset_stats": dataset.summary(),
        "split_sizes": report["sizes"],
        "split_time_ranges": {
            "train": list(report["train_range"]),
            "valid": list(report["valid_range"]),
            "test": list(report["test_range"]),
        },
        "strictly_ordered": report["strictly_ordered"],
        "items_unseen_in_train": report["items_unseen_in_train"],
        "negative_pool_size": len(pool),
        "negative_pool_cutoff": train_max_ts,
        "eval_num_neg": eval_num_neg,
        "sequence_stats": stats,
        "config": {
            "max_len": max_len,
            "min_len": min_len,
        },
    }

    if args.dry_run:
        logger.info("--dry-run: 跳过文件写入")
        logger.info("统计摘要:\n%s", json.dumps(summary, ensure_ascii=False, indent=2)[:2000])
        return 0

    # --- 7. 导出 ---
    processed_dir.mkdir(parents=True, exist_ok=True)
    export_interactions(split, processed_dir)
    export_sequences(sequences, processed_dir)
    export_eval_negatives(eval_negatives, processed_dir)
    export_metadata(dataset, processed_dir)
    export_summary(summary, processed_dir)

    logger.info("预处理完成 -> %s", processed_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
