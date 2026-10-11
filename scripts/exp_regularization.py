"""S1-W3：过拟合与正则化对照实验。

## 实验设计

要验证正则化是否有效，前提是**先造出明显的过拟合**。所以基线配置刻意偏向过拟合：
- 只用户子集（默认 1500 人）→ 训练数据少
- embedding_dim = 128 → 容量大（相对于数据量）
- 固定训练 12 轮，不用早停 → 让过拟合充分暴露

然后逐个加入正则化手段，看哪个能压缩 train/valid 的差距。

## 四个核心指标

| 指标 | 含义 | 期望方向 |
| --- | --- | --- |
| `train_loss` | 训练集损失 | 降 |
| `valid_ndcg@10` | 验证集排序质量 | **升**（这是唯一真正重要的） |
| `gap_loss` | valid 末轮 loss − train 末轮 loss | **降**（正则化是否起效） |
| `best_epoch` | 最优轮次 | 变大说明训练更持久（没早早过拟合） |

`gap` 是本实验的核心度量：**它衡量泛化差距**。
无正则时 gap 会随轮数扩大；正则化生效时 gap 被压住。

用法::

    python scripts/exp_regularization.py
    python scripts/exp_regularization.py --users 1500 --epochs 12 --batch-size 256
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recagent.data.dataset import SequenceRecallDataset, load_sequences
from recagent.data.negative import NegativeSampler
from recagent.models.sequential import SequentialRecallModel
from recagent.train import TrainConfig, Trainer
from recagent.utils.logging import get_logger
from recagent.utils.seeding import seed_everything

logger = get_logger("exp_regularization")


@dataclass
class ExperimentConfig:
    """一个实验配置。名字会直接写进结果表，必须能自解释。"""

    name: str
    embedding_dim: int = 128
    hidden_dim: int = 256
    dropout: float = 0.0
    weight_decay: float = 0.0
    lr: float = 1e-3
    train_neg: int = 99
    note: str = ""


# 对照组设计：每次只改一个维度，才能归因
EXPERIMENTS: list[ExperimentConfig] = [
    # --- 基线：故意过拟合 ---
    ExperimentConfig("baseline_no_reg", note="无正则，容量大 + 数据少"),
    # --- 容量对照：证明"过拟合来自容量超过数据量" ---
    ExperimentConfig("capacity_dim32", embedding_dim=32, note="只降容量，不加正则"),
    ExperimentConfig("capacity_nomlp", hidden_dim=0, note="去掉 MLP 头，回到纯内积"),
    # --- weight decay 扫描 ---
    ExperimentConfig("wd_1e-3", weight_decay=1e-3, note="轻正则"),
    ExperimentConfig("wd_1e-2", weight_decay=1e-2, note="中等正则"),
    ExperimentConfig("wd_1e-1", weight_decay=1e-1, note="强正则"),
    # --- dropout 扫描 ---
    ExperimentConfig("drop_0.2", dropout=0.2, note="轻 dropout"),
    ExperimentConfig("drop_0.5", dropout=0.5, note="强 dropout"),
    # --- 组合 ---
    ExperimentConfig(
        "combo_wd1e-2_drop0.3",
        dropout=0.3,
        weight_decay=1e-2,
        note="组合正则",
    ),
]


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in batch]) for key in batch[0]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="过拟合与正则化对照实验")
    parser.add_argument("--data-dir", default="data/processed/ml1m")
    parser.add_argument("--users", type=int, default=2500, help="采样多少用户（造过拟合）")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-neg", type=int, default=99)
    parser.add_argument("--max-seq-len", type=int, default=50)
    parser.add_argument("--min-prefix", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", default=None, help="只跑名字包含该子串的实验")
    parser.add_argument("--output-dir", default="outputs/exp_regularization")
    return parser


def load_data(data_dir: Path) -> tuple[Any, ...]:
    sequences = load_sequences(data_dir / "sequences_train.npz")
    train = pd.read_csv(data_dir / "interactions_train.csv")
    valid = pd.read_csv(data_dir / "interactions_valid.csv")

    valid_positives: dict[int, list[int]] = (
        valid.groupby("user_id")["item_id"].apply(list).to_dict()
    )
    return sequences, train, valid_positives


def subsample(
    sequences: dict[int, list[int]],
    users: int,
    valid_users: set[int],
    desired_evaluable: int,
    seed: int,
) -> dict[int, list[int]]:
    """采样用户子集，优先选中"有验证正样本"的用户。

    为什么需要这个偏好：验证指标只在"该用户既有训练序列、又有 valid 正样本"
    时才产生贡献。纯随机采样会让有效验证用户极少（实测 1200 用户里只有 117 个），
    valid_ndcg 的噪声会大到无法比较不同正则化手段。

    做法：先从可评测用户里抽 80%，其余用其他用户补齐。
    仍是随机采样（同 seed 可复现），只是约束了构成。
    """
    all_users = sorted(sequences)
    if users >= len(all_users):
        return sequences

    rng = np.random.default_rng(seed)
    evaluable = [user for user in all_users if user in valid_users]
    other = [user for user in all_users if user not in valid_users]

    target = min(len(evaluable), desired_evaluable)
    picked: set[int] = set()
    if target > 0:
        picked.update(evaluable[i] for i in rng.choice(len(evaluable), size=target, replace=False))

    remaining = users - len(picked)
    if remaining > 0 and other:
        picked.update(
            other[i] for i in rng.choice(len(other), size=min(remaining, len(other)), replace=False)
        )

    return {user: sequences[user] for user in all_users if user in picked}


def run_experiment(
    experiment: ExperimentConfig,
    *,
    sequences: dict[int, list[int]],
    train: pd.DataFrame,
    valid_positives: dict[int, list[int]],
    args: argparse.Namespace,
    negative_pool: np.ndarray,
) -> dict[str, Any]:
    """跑一个配置，返回该配置的汇总指标。"""
    seed_everything(args.seed)

    users = sorted(sequences)
    num_items = int(train["item_id"].max())

    sampler = NegativeSampler(negative_pool.tolist(), seed=args.seed)
    val_negatives = sampler.eval_negatives(users, num_neg=args.eval_neg)

    # 训练集：只用采样到用户的交互（否则数据量没变，造不出过拟合）
    user_set = set(users)
    train_subset = train[train["user_id"].isin(user_set)]

    train_dataset = SequenceRecallDataset(
        sequences=sequences,
        positives={},
        negatives=None,
        num_items=num_items,
        max_seq_len=args.max_seq_len,
        num_neg=experiment.train_neg,
        negative_pool=negative_pool,
        seed=args.seed,
        min_prefix=args.min_prefix,
        mode="train",
        # 固定训练负样本：让训练目标与"固定候选集"的评测协议一致。
        # 不固定的话 train_loss 降而 valid_ndcg 也降，评测信号失效（详见 dataset.py 注释）。
        fixed_negatives=True,
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate
    )

    model = SequentialRecallModel(
        num_items=num_items,
        embedding_dim=experiment.embedding_dim,
        hidden_dim=experiment.hidden_dim,
        dropout=experiment.dropout,
    )
    num_params = sum(p.numel() for p in model.parameters())

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=experiment.lr,
        # **必须用 AdamW**。这是本实验踩过的真实坑：
        # 用 Adam + weight_decay 时（L2 路径），embedding 层大多数行当步没有梯度，
        # 却仍被施加 L2 衰减，衰减速度超过稀疏梯度更新速度 → 整个 embedding 被压向 0，
        # 模型输出恒为 0，train_loss 卡在 ln(2)=0.6931。
        # AdamW 把 weight decay 解耦出梯度，不会伤害稀疏参数。
        optimizer="adamw",
        weight_decay=experiment.weight_decay,
        grad_clip=5.0,
        eval_every_n_epochs=1,
        early_stop_patience=None,  # 关掉早停，让所有配置跑满相同轮数，公平对比
        monitor="ndcg@10",
        mode="max",
        log_every_n_steps=0,
        device=args.device,
        seed=args.seed,
    )

    # 不传任何回调：本实验只用 Trainer 的记录能力，不需要存 checkpoint
    trainer = Trainer(model, train_loader, valid_loader, config, [])
    history = trainer.fit()

    train_losses = [r["loss"] for r in history.train_metrics]
    valid_losses = [r["loss"] for r in history.valid_metrics]
    valid_ndcg = [r["ndcg@10"] for r in history.valid_metrics]

    best_epoch = int(np.argmax(valid_ndcg))
    return {
        "name": experiment.name,
        "note": experiment.note,
        "embedding_dim": experiment.embedding_dim,
        "hidden_dim": experiment.hidden_dim,
        "dropout": experiment.dropout,
        "weight_decay": experiment.weight_decay,
        "num_params": num_params,
        "train_samples": len(train_dataset),
        "interactions": len(train_subset),
        "train_loss_first": train_losses[0],
        "train_loss_last": train_losses[-1],
        "valid_loss_last": valid_losses[-1],
        # gap 是本实验的核心度量：泛化差距
        "gap_loss": valid_losses[-1] - train_losses[-1],
        "valid_ndcg_first": valid_ndcg[0],
        "valid_ndcg_best": float(max(valid_ndcg)),
        "valid_ndcg_last": valid_ndcg[-1],
        "best_epoch": best_epoch,
        # 相对首轮的提升
        "ndcg_gain_pct": (max(valid_ndcg) / max(valid_ndcg[0], 1e-9) - 1) * 100,
        # 过拟合程度：最优之后退步了多少
        "ndcg_decay_after_best_pct": (
            (max(valid_ndcg) - valid_ndcg[-1]) / max(max(valid_ndcg), 1e-9) * 100
        ),
        "epochs": args.epochs,
    }


def print_table(results: list[dict[str, Any]]) -> None:
    header = (
        f"{'配置':<24}{'参数':>9}{'train_loss':>12}{'gap':>9}"
        f"{'ndcg首':>9}{'ndcg最优':>10}{'最优轮':>7}{'退步%':>8}"
    )
    print()
    print("=" * len(header))
    print("过拟合与正则化对照结果")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{row['name']:<24}{row['num_params']:>9,}"
            f"{row['train_loss_last']:>12.4f}{row['gap_loss']:>9.4f}"
            f"{row['valid_ndcg_first']:>9.4f}{row['valid_ndcg_best']:>10.4f}"
            f"{row['best_epoch']:>7}{row['ndcg_decay_after_best_pct']:>8.1f}"
        )
    print("-" * len(header))

    baseline = next((r for r in results if r["name"] == "baseline_no_reg"), None)
    if baseline is not None:
        print()
        print("相对基线的变化（正数 = 更好）")
        print(f"{'配置':<24}{'Δndcg最优':>12}{'Δgap':>10}{'判定':>12}")
        for row in results:
            if row["name"] == baseline["name"]:
                continue
            delta_ndcg = row["valid_ndcg_best"] - baseline["valid_ndcg_best"]
            delta_gap = row["gap_loss"] - baseline["gap_loss"]
            verdict = (
                "有效"
                if (delta_gap < 0 and delta_ndcg > 0)
                else ("只压 gap" if delta_gap < 0 else "无效/更差")
            )
            print(f"{row['name']:<24}{delta_ndcg:>+12.4f}{delta_gap:>+10.4f}{verdict:>12}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        logger.error("预处理目录不存在: %s", data_dir)
        return 2

    sequences_all, train_all, valid_positives_all = load_data(data_dir)
    valid_user_set = set(valid_positives_all)
    # 需要多少个"可评测用户"（既有训练序列、又有 valid 正样本）。
    # 若采样子集和验证集不匹配，验证指标衡量的就不是泛化误差，
    # 而是"训练分布 vs 评测分布"的差异 —— 正则化对此无能为力。
    needed_evaluable = max(1, int(args.users * 0.8))
    sequences = subsample(sequences_all, args.users, valid_user_set, needed_evaluable, args.seed)

    # 验证集也要限制在采样子集内，否则"训练少、验证多"会让 gap 失去意义
    user_set = set(sequences)
    valid_positives = {
        user: items for user, items in valid_positives_all.items() if user in user_set
    }

    negative_pool = np.unique(train_all["item_id"].to_numpy())

    logger.info(
        "采样子集: %s 用户（原 %s）| 训练交互 %s 条 | 验证用户 %s",
        len(sequences),
        len(sequences_all),
        int(train_all["user_id"].isin(user_set).sum()),
        len(valid_positives),
    )

    experiments = EXPERIMENTS
    if args.only:
        experiments = [e for e in EXPERIMENTS if args.only in e.name]
        if not experiments:
            logger.error("没有匹配 %r 的实验", args.only)
            return 2

    results: list[dict[str, Any]] = []
    for index, experiment in enumerate(experiments, start=1):
        logger.info("=" * 70)
        logger.info("[%s/%s] %s —— %s", index, len(experiments), experiment.name, experiment.note)
        logger.info("=" * 70)
        row = run_experiment(
            experiment,
            sequences=sequences,
            train=train_all,
            valid_positives=valid_positives,
            args=args,
            negative_pool=negative_pool,
        )
        results.append(row)
        logger.info(
            "  -> train_loss=%.4f gap=%.4f ndcg最优=%.4f（首轮 %.4f, 第 %s 轮）",
            row["train_loss_last"],
            row["gap_loss"],
            row["valid_ndcg_best"],
            row["valid_ndcg_first"],
            row["best_epoch"],
        )

    print_table(results)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": vars(args),
        "experiments": [asdict(e) for e in experiments],
        "results": results,
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("结果已存 -> %s", output_dir / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
