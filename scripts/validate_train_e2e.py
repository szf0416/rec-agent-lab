"""S1 收尾：用真实 ML-1M 数据端到端验证训练循环。

目的不是刷指标，而是证明 S1 写的训练基础设施**在真实数据上真的能用**：
- batch 结构（嵌套张量）能正确搬到 GPU
- Dataset/DataLoader 与 Trainer 的接口对得上
- 损失能下降、排序指标能提升（说明学到东西，不是空转）
- checkpoint 能存能读

用法::

    python scripts/validate_train_e2e.py
    python scripts/validate_train_e2e.py --epochs 5 --batch-size 256
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recagent.data.dataset import SequenceRecallDataset, load_sequences
from recagent.models.sequential import SequentialRecallModel
from recagent.train import (
    EarlyStopping,
    MLflowLogger,
    ModelCheckpoint,
    SchedulerConfig,
    TrainConfig,
    Trainer,
)
from recagent.utils.logging import get_logger
from recagent.utils.seeding import seed_everything

logger = get_logger("validate_e2e")


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """默认 collate 无法处理 dict 列表，需要手动堆叠。

    训练循环要求 batch 是「张量的嵌套结构」，这个函数负责把
    DataLoader 给的 list-of-dict 转成 dict-of-tensor。
    """
    return {key: torch.stack([item[key] for item in batch]) for key in batch[0]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="真实数据端到端验证")
    parser.add_argument("--data-dir", default="data/processed/ml1m")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-seq-len", type=int, default=50)
    parser.add_argument("--min-prefix", type=int, default=20, help="训练样本最短前缀")
    parser.add_argument("--train-neg", type=int, default=4, help="训练负样本数")
    parser.add_argument("--eval-neg", type=int, default=99, help="评测负样本数")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--output-dir", default="outputs/e2e_validate")
    return parser


def load_data(data_dir: Path) -> tuple[Any, ...]:
    """读取预处理产物。"""
    sequences = load_sequences(data_dir / "sequences_train.npz")

    train = pd.read_csv(data_dir / "interactions_train.csv")
    valid = pd.read_csv(data_dir / "interactions_valid.csv")
    test = pd.read_csv(data_dir / "interactions_test.csv")

    # 正样本：验证/测试集里每位用户的交互物品
    valid_positives: dict[int, list[int]] = (
        valid.groupby("user_id")["item_id"].apply(list).to_dict()
    )
    test_positives: dict[int, list[int]] = test.groupby("user_id")["item_id"].apply(list).to_dict()

    return sequences, train, valid_positives, test_positives


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        logger.error("预处理目录不存在: %s", data_dir)
        logger.error("请先运行: python scripts/preprocess.py --config configs/data/ml1m.yaml")
        return 2

    seed_everything(args.seed)
    sequences, train, valid_positives, test_positives = load_data(data_dir)

    # 物品上限：embedding 表大小取 max(item_id) + 1（id 从 1 开始，0 留给 padding）
    # 注意不能只看训练集：验证/测试集里可能出现训练期不存在的新物品，
    # embedding 表若按训练集上界建，遇到新物品会索引越界。
    max_item_id = int(train["item_id"].max())
    for positives in (valid_positives, test_positives):
        for items in positives.values():
            if items:
                max_item_id = max(max_item_id, max(items))
    num_items = max_item_id

    logger.info(
        "序列 %s 用户 | 训练交互 %s 条 | 最大物品 id %s", len(sequences), len(train), num_items
    )

    # ----------------------------------------------------------------
    # 负采样候选池：只取训练集里的物品
    # ----------------------------------------------------------------
    # 时间切分下，训练集里的物品必然在训练期已存在，所以直接用它当池子
    # 就自动满足"不用未来物品做负样本"的约束，不需要额外按时间戳过滤。
    negative_pool = np.unique(train["item_id"].to_numpy())
    logger.info("负采样候选池: %s 个物品（全部来自训练集）", len(negative_pool))

    # ----------------------------------------------------------------
    # 数据集
    # ----------------------------------------------------------------
    # 负采样协议：为 valid 和 test 各自生成一套**固定**负样本。
    #
    # 为什么必须固定：若每次评测重新随机采样，两个模型就是在不同题集上考试，
    # 指标差异里混入采样噪声，模型之间不可比（这是 S3 做消融的前提）。
    #
    # 为什么 val 和 test 要分开：用 test 做早停属于"拿考试题调参"，
    # 最终报告的 test 指标会偏乐观。val 早停、test 只报告一次，才是干净的。
    from recagent.data.negative import NegativeSampler

    # 候选池就用训练集出现过的物品：时间切分保证它们都在训练期已存在，
    # 天然满足"不用未来物品做负样本"，不需要再按时间戳过滤。
    sampler = NegativeSampler(negative_pool.tolist(), seed=args.seed)

    val_users = sorted(valid_positives)
    test_users = sorted(test_positives)
    val_negatives = sampler.eval_negatives(val_users, num_neg=args.eval_neg)
    test_negatives = sampler.eval_negatives(test_users, num_neg=args.eval_neg)
    logger.info(
        "固定负样本: val %s 用户 / test %s 用户（每人 %s 个）",
        len(val_negatives),
        len(test_negatives),
        args.eval_neg,
    )

    train_dataset = SequenceRecallDataset(
        sequences=sequences,
        positives={},
        negatives=None,
        num_items=num_items,
        max_seq_len=args.max_seq_len,
        num_neg=args.train_neg,
        negative_pool=negative_pool,
        seed=args.seed,
        min_prefix=args.min_prefix,
        mode="train",
    )

    valid_dataset = SequenceRecallDataset(
        sequences=sequences,
        positives=valid_positives,
        negatives=val_negatives,
        num_items=num_items,
        max_seq_len=args.max_seq_len,
        num_neg=args.eval_neg,
        seed=args.seed,
        mode="eval",
    )

    test_dataset = SequenceRecallDataset(
        sequences=sequences,
        positives=test_positives,
        negatives=test_negatives,
        num_items=num_items,
        max_seq_len=args.max_seq_len,
        num_neg=args.eval_neg,
        seed=args.seed,
        mode="eval",
    )

    logger.info(
        "训练样本 %s | 验证样本 %s | 测试样本 %s",
        len(train_dataset),
        len(valid_dataset),
        len(test_dataset),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    # ----------------------------------------------------------------
    # 模型与训练
    # ----------------------------------------------------------------
    model = SequentialRecallModel(num_items=num_items, embedding_dim=args.embedding_dim)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info("模型参数量: %s", f"{num_params:,}")

    output_dir = Path(args.output_dir)
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        optimizer="adamw",
        weight_decay=0.01,
        grad_clip=5.0,
        scheduler=SchedulerConfig(name="cosine", interval="epoch", warmup_steps=50),
        early_stop_patience=5,
        early_stop_restore_best=True,
        monitor="ndcg@10",
        mode="max",
        eval_every_n_epochs=1,
        log_every_n_steps=0,
        device=args.device,
        seed=args.seed,
        checkpoint_dir=str(output_dir / "checkpoints"),
    )

    callbacks: list[Any] = [
        ModelCheckpoint(output_dir / "checkpoints", monitor="ndcg@10", mode="max", save_top_k=1),
        EarlyStopping("ndcg@10", patience=5, mode="max", restore_best=True),
    ]
    if not args.no_mlflow:
        callbacks.append(
            MLflowLogger(experiment_name="recagent-e2e", run_name="seq-recall-baseline")
        )

    trainer = Trainer(model, train_loader, valid_loader, config, callbacks)
    history = trainer.fit()

    # ----------------------------------------------------------------
    # 最终评测：用 test 集报告一次
    #
    # 为什么不在训练中途看 test：那是"拿考试题调参"，指标会偏乐观。
    # 早停只看 val，test 只在最后报告一次。
    #
    # 注意：EarlyStopping(restore_best=True) 已经把权重恢复成 val 最优的那一轮，
    # 所以这里评测的就是"val 上最好的模型"。
    # ----------------------------------------------------------------
    test_trainer = Trainer(model, train_loader, test_loader, config, [])
    test_metrics = test_trainer.validate()
    logger.info("=" * 70)
    logger.info(
        "TEST 最终指标: hr@10=%.4f | ndcg@10=%.4f | mrr@10=%.4f | loss=%.4f",
        test_metrics.get("hr@10", float("nan")),
        test_metrics.get("ndcg@10", float("nan")),
        test_metrics.get("mrr@10", float("nan")),
        test_metrics.get("loss", float("nan")),
    )
    logger.info("=" * 70)

    # ----------------------------------------------------------------
    # 结果
    # ----------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("端到端验证结果")
    logger.info("=" * 70)
    logger.info("%-6s %-12s %-10s %-10s %-10s", "epoch", "train_loss", "hr@10", "ndcg@10", "mrr@10")
    for epoch, train_record in enumerate(history.train_metrics):
        valid_record = history.valid_metrics[epoch] if epoch < len(history.valid_metrics) else {}
        logger.info(
            "%-6s %-12.4f %-10.4f %-10.4f %-10.4f",
            epoch,
            train_record["loss"],
            valid_record.get("hr@10", float("nan")),
            valid_record.get("ndcg@10", float("nan")),
            valid_record.get("mrr@10", float("nan")),
        )

    first_ndcg = history.valid_metrics[0]["ndcg@10"]
    best_ndcg = max(record["ndcg@10"] for record in history.valid_metrics)
    logger.info("-" * 70)
    logger.info(
        "首轮 ndcg@10 = %.4f | 最优 = %.4f | 相对提升 = %+.1f%%",
        first_ndcg,
        best_ndcg,
        (best_ndcg / max(first_ndcg, 1e-9) - 1) * 100,
    )
    logger.info(
        "训练轮数 = %s | 早停 = %s | 最优 epoch = %s",
        history.num_epochs,
        history.stopped_early,
        history.best_epoch,
    )

    # 判定标准要宽松：轮数很少时（如 2 轮）"第 0 轮即最优"是正常的，
    # 此时 best == first。要求严格提升会误报失败，让人以为流程坏了。
    decreased = history.train_metrics[-1]["loss"] < history.train_metrics[0]["loss"]
    improved = best_ndcg >= first_ndcg
    logger.info("loss 下降 = %s | 指标未退化 = %s", decreased, improved)

    # 保存一次结果，便于写进 reports/
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "history.json").write_text(
        json.dumps(history.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("训练历史已存 -> %s", output_dir / "history.json")

    return 0 if (decreased and improved) else 1


if __name__ == "__main__":
    raise SystemExit(main())
