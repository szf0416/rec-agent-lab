"""S1-W4：优化器与超参对照实验。

## 四个模式

| 模式 | 回答什么问题 |
| --- | --- |
| `optimizers` | Adam / AdamW / SGD / SGD+Momentum 差在哪？`Adam+L2` 与 `AdamW` 有什么不同？ |
| `lr_range` | 学习率该取多少？lr range test 怎么读？ |
| `warmup` | 预热到底有没有用？多少步合适？ |
| `batch_size` | 大 batch 要不要按比例放大 lr？ |

## 为什么这批实验的重点是「看曲线」

S1-W1~W3 你已经掌握了机制，但暴露的短板是**从现象反推原因**。
所以本脚本的每一项都输出「曲线长什么样」而不只是最终指标。

## 复用

数据加载 / 采样 / collate 复用 W3 的实现（只读引用，不改动 W3 脚本）。

用法::

    python scripts/exp_optimizer.py --mode optimizers
    python scripts/exp_optimizer.py --mode lr_range
    python scripts/exp_optimizer.py --mode warmup
    python scripts/exp_optimizer.py --mode batch_size
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recagent.data.dataset import SequenceRecallDataset
from recagent.data.negative import NegativeSampler
from recagent.models.sequential import SequentialRecallModel
from recagent.train import TrainConfig, Trainer
from recagent.utils.logging import get_logger
from recagent.utils.seeding import seed_everything
from scripts.exp_regularization import collate, load_data, subsample

logger = get_logger("exp_optimizer")


@dataclass
class OptConfig:
    """一个优化相关的实验配置。"""

    name: str
    optimizer: str = "adamw"
    lr: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 0.0
    batch_size: int = 256
    warmup_steps: int = 0
    epochs: int = 0  # 0 表示用命令行传入的轮数
    note: str = ""


# --------------------------------------------------------------------------
# 模式 1：优化器对比
# --------------------------------------------------------------------------

OPTIMIZER_EXPERIMENTS: list[OptConfig] = [
    OptConfig("adam", optimizer="adam", note="基线：Adam 无正则"),
    OptConfig("adamw", optimizer="adamw", note="基线：AdamW 无正则"),
    # ↓ 这两行是本模式的核心对照：同样的 weight_decay，不同路径
    OptConfig("adam_l2_wd0.01", optimizer="adam", weight_decay=0.01, note="Adam 的 L2 路径"),
    OptConfig("adamw_wd0.01", optimizer="adamw", weight_decay=0.01, note="AdamW 的解耦衰减"),
    # SGD 需要显著更大的 lr，否则几乎不动。
    #
    # 必须显式写 momentum=0.0：OptConfig 的 momentum 默认值是 0.9，
    # 不写的话 "sgd" 与 "sgd_momentum" 会变成完全相同的配置。
    # 第一次跑时两者 train_loss / ndcg / 最优轮 一模一样，就是这个原因。
    # 教训：两个配置结果**完全相同**时，先怀疑配置重复，别当成实验噪声。
    OptConfig("sgd", optimizer="sgd", momentum=0.0, lr=0.05, note="SGD 无动量(momentum=0)"),
    OptConfig("sgd_momentum", optimizer="sgd", momentum=0.9, lr=0.05, note="SGD + 动量"),
]

# --------------------------------------------------------------------------
# 模式 3：warmup 对比
# --------------------------------------------------------------------------

WARMUP_EXPERIMENTS: list[OptConfig] = [
    OptConfig("warmup_0", warmup_steps=0, note="无预热"),
    OptConfig("warmup_100", warmup_steps=100, note="短预热"),
    OptConfig("warmup_500", warmup_steps=500, note="长预热"),
]

# --------------------------------------------------------------------------
# 模式 4：batch size 与 lr 的关系
# --------------------------------------------------------------------------

BATCH_EXPERIMENTS: list[OptConfig] = [
    OptConfig("bs128_lr1e-3", batch_size=128, lr=1e-3, note="基线"),
    OptConfig("bs512_lr1e-3", batch_size=512, lr=1e-3, note="只放大 batch"),
    OptConfig("bs512_lr4e-3", batch_size=512, lr=4e-3, note="batch ×4，lr 也 ×4（线性缩放）"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="优化器与超参对照实验")
    parser.add_argument(
        "--mode",
        default="optimizers",
        choices=["optimizers", "lr_range", "warmup", "batch_size"],
    )
    parser.add_argument("--data-dir", default="data/processed/ml1m")
    parser.add_argument("--users", type=int, default=1800)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eval-neg", type=int, default=99)
    parser.add_argument("--train-neg", type=int, default=99)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=50)
    parser.add_argument("--min-prefix", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", default=None, help="只跑名字包含该子串的实验")
    parser.add_argument("--output-dir", default="outputs/exp_optimizer")
    return parser


class ExperimentRunner:
    """把数据准备一次，重复用于多个配置。

    为什么做成类：数据加载 + 采样 + 负采样约 5 秒，
    跑 6 个配置就浪费 30 秒。准备一次即可。
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        data_dir = Path(args.data_dir)
        sequences_all, train_all, valid_positives_all = load_data(data_dir)

        valid_users = set(valid_positives_all)
        needed = max(1, int(args.users * 0.8))
        self.sequences = subsample(sequences_all, args.users, valid_users, needed, args.seed)
        user_set = set(self.sequences)
        self.valid_positives = {
            user: items for user, items in valid_positives_all.items() if user in user_set
        }

        self.train_all = train_all
        self.negative_pool = np.unique(train_all["item_id"].to_numpy())
        self.num_items = int(train_all["item_id"].max())
        self.user_set = user_set

        sampler = NegativeSampler(self.negative_pool.tolist(), seed=args.seed)
        self.val_negatives = sampler.eval_negatives(sorted(self.sequences), num_neg=args.eval_neg)

        logger.info(
            "数据就绪: %s 用户 | %s 训练交互 | %s 验证用户 | 候选池 %s",
            len(self.sequences),
            int(train_all["user_id"].isin(user_set).sum()),
            len(self.valid_positives),
            len(self.negative_pool),
        )

    def run(
        self, config: OptConfig, *, epochs: int | None = None, track: bool = False
    ) -> dict[str, Any]:
        """跑一个配置，返回汇总指标。"""
        args = self.args
        seed_everything(args.seed)

        train_dataset = SequenceRecallDataset(
            sequences=self.sequences,
            positives={},
            negatives=None,
            num_items=self.num_items,
            max_seq_len=args.max_seq_len,
            num_neg=args.train_neg,
            negative_pool=self.negative_pool,
            seed=args.seed,
            min_prefix=args.min_prefix,
            mode="train",
            fixed_negatives=True,
        )
        valid_dataset = SequenceRecallDataset(
            sequences=self.sequences,
            positives=self.valid_positives,
            negatives=self.val_negatives,
            num_items=self.num_items,
            max_seq_len=args.max_seq_len,
            num_neg=args.eval_neg,
            seed=args.seed,
            mode="eval",
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate,
            drop_last=True,
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate
        )

        model = SequentialRecallModel(
            num_items=self.num_items,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            dropout=0.0,
        )

        total_epochs = epochs if epochs is not None else (config.epochs or args.epochs)
        train_config = TrainConfig(
            epochs=total_epochs,
            batch_size=config.batch_size,
            lr=config.lr,
            optimizer=config.optimizer,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            grad_clip=5.0,
            eval_every_n_epochs=1,
            early_stop_patience=None,  # 关掉，让所有配置跑满相同轮数才公平
            monitor="ndcg@10",
            mode="max",
            log_every_n_steps=0,
            device=args.device,
            seed=args.seed,
        )
        if config.warmup_steps > 0:
            from recagent.train import SchedulerConfig

            train_config.scheduler = SchedulerConfig(
                name="none", interval="step", warmup_steps=config.warmup_steps
            )

        trainer = Trainer(model, train_loader, valid_loader, train_config, [])
        history = trainer.fit()

        train_losses = [r["loss"] for r in history.train_metrics]
        valid_losses = [r["loss"] for r in history.valid_metrics]
        valid_ndcg = [r["ndcg@10"] for r in history.valid_metrics]
        grad_norms = history.grad_norm_history or [float("nan")]

        return {
            "name": config.name,
            "note": config.note,
            "optimizer": config.optimizer,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "batch_size": config.batch_size,
            "warmup_steps": config.warmup_steps,
            "epochs": total_epochs,
            "train_loss_first": train_losses[0],
            "train_loss_last": train_losses[-1],
            "valid_loss_last": valid_losses[-1],
            "gap_loss": valid_losses[-1] - train_losses[-1],
            "valid_ndcg_first": valid_ndcg[0],
            "valid_ndcg_best": float(max(valid_ndcg)),
            "valid_ndcg_last": valid_ndcg[-1],
            "best_epoch": int(np.argmax(valid_ndcg)),
            "grad_norm_first": grad_norms[0],
            "grad_norm_mean": float(np.nanmean(grad_norms)),
            # 曲线形状判据：前期是否上升过
            "ndcg_ever_rose": max(valid_ndcg) > valid_ndcg[0] * 1.001,
            "_train_losses": train_losses,
            "_valid_ndcg": valid_ndcg,
            "_step_lr": history.step_lr_history if track else [],
        }


# --------------------------------------------------------------------------
# lr range test
# --------------------------------------------------------------------------


def run_lr_range(runner: ExperimentRunner, args: argparse.Namespace) -> dict[str, Any]:
    """学习率扫描：lr 从极小按指数增长，观察 loss 的变化。

    ## 怎么读这条曲线

    ```
    loss
     │＼
     │  ＼___          ← 下降最快的区域 = 可用 lr
     │      ＼___
     │          ＼___  ← 开始上升/震荡 = lr 过大
     └──────────────────▶ lr（对数轴）
     1e-6        1e-1
    ```

    - **loss 下降最快的斜率处** → 好的学习率量级
    - **loss 开始上升或震荡处** → 上界，不能超过
    - 实际取值通常比"最优斜率点"小一个量级，留安全边际

    实现：用 `LambdaLR` 让 lr 指数增长，每个 step 记录 (lr, loss)。
    注意每个 step 只跑一个 batch —— 这是 range test 的标准做法，
    目的是**快速扫过 lr 空间**，不是真训练。
    """
    from torch.optim.lr_scheduler import LambdaLR

    seed_everything(args.seed)

    train_dataset = SequenceRecallDataset(
        sequences=runner.sequences,
        positives={},
        negatives=None,
        num_items=runner.num_items,
        max_seq_len=args.max_seq_len,
        num_neg=args.train_neg,
        negative_pool=runner.negative_pool,
        seed=args.seed,
        min_prefix=args.min_prefix,
        mode="train",
        fixed_negatives=True,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
        drop_last=True,
    )

    model = SequentialRecallModel(
        num_items=runner.num_items,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=0.0,
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    lr_min, lr_max = 1e-6, 3e-1
    # base_lr 必须是 lr_min：LambdaLR 的因子是**乘法**的，
    # 实际 lr = base_lr × factor。若 base_lr 设成 1.0，实际 lr 会跑到
    # 1.0 × (lr_max/lr_min) = 3e5，完全偏离扫描区间 ——
    # 第一次跑就出现了 lr=4.16e+01 这种越界值（超过扫描上界 0.3 一百倍）。
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_min)
    total_steps = min(600, len(loader) * 2)
    # factor 从 1 增长到 lr_max/lr_min，于是 lr 从 lr_min 增长到 lr_max
    growth = math.log(lr_max / lr_min)
    scheduler = LambdaLR(
        optimizer, lr_lambda=lambda step: math.exp(growth * step / max(total_steps - 1, 1))
    )

    records: list[dict[str, float]] = []
    model.train()
    step = 0
    smoothed_loss = 0.0
    while step < total_steps:
        for batch in loader:
            if step >= total_steps:
                break
            moved = {
                key: value.to(next(model.parameters()).device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            loss = model.forward(moved)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            current_lr = float(optimizer.param_groups[0]["lr"])
            scheduler.step()

            loss_value = float(loss.detach().item())
            # 指数滑动平均：单 batch 的 loss 噪声很大，不平滑根本看不出趋势
            smoothed_loss = loss_value if step == 0 else 0.9 * smoothed_loss + 0.1 * loss_value
            records.append(
                {"step": step, "lr": current_lr, "loss": loss_value, "smooth_loss": smoothed_loss}
            )
            step += 1

    # 找"平滑 loss 下降最快"的位置（用负斜率最大的窗口）
    lrs = np.array([r["lr"] for r in records])
    smooth = np.array([r["smooth_loss"] for r in records])
    window = max(10, len(records) // 30)
    slopes = np.full(len(records), np.nan)
    for i in range(window, len(records)):
        slopes[i] = (smooth[i] - smooth[i - window]) / window
    best_idx = int(np.nanargmin(slopes))

    # 上界：平滑 loss 开始持续上升的位置
    min_idx = int(np.nanargmin(smooth))
    upper = lrs[min(min_idx + window, len(lrs) - 1)]

    return {
        "mode": "lr_range",
        "lr_min": lr_min,
        "lr_max": lr_max,
        "total_steps": total_steps,
        "steepest_descent_lr": float(lrs[best_idx]),
        "steepest_descent_loss": float(smooth[best_idx]),
        "loss_minimum_lr": float(lrs[min_idx]),
        "suggested_lr_upper_bound": float(upper),
        "records": records,
    }


# --------------------------------------------------------------------------
# 输出
# --------------------------------------------------------------------------


def print_table(results: list[dict[str, Any]]) -> None:
    header = (
        f"{'配置':<20}{'opt':>8}{'lr':>8}{'bs':>6}"
        f"{'train_loss':>12}{'gap':>9}{'ndcg最优':>10}{'最优轮':>7}"
    )
    print()
    print("=" * len(header))
    print("优化器与超参对照")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{row['name']:<20}{row['optimizer']:>8}{row['lr']:>8.0e}{row['batch_size']:>6}"
            f"{row['train_loss_last']:>12.4f}{row['gap_loss']:>9.4f}"
            f"{row['valid_ndcg_best']:>10.4f}{row['best_epoch']:>7}"
        )
    print("-" * len(header))

    # 曲线形状预警：train_loss 是否"卡住"
    stuck = [r for r in results if abs(r["train_loss_last"] - r["train_loss_first"]) < 1e-4]
    if stuck:
        print()
        print("⚠️  以下配置的 train_loss 几乎没变（可能没在学）：")
        for row in stuck:
            print(
                f"    {row['name']}: {row['train_loss_first']:.4f} -> "
                f"{row['train_loss_last']:.4f}"
                + (
                    "   ← 恰好是 ln(2)=0.6931，疑似输出恒为 0"
                    if abs(row["train_loss_first"] - math.log(2)) < 1e-3
                    else ""
                )
            )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not Path(args.data_dir).is_dir():
        logger.error("预处理目录不存在: %s", args.data_dir)
        return 2

    runner = ExperimentRunner(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- lr_range 是独立流程 ---
    if args.mode == "lr_range":
        logger.info("=" * 70)
        logger.info("学习率扫描（lr range test）")
        logger.info("=" * 70)
        result = run_lr_range(runner, args)
        print()
        print("=" * 70)
        print("lr range test 结果")
        print("=" * 70)
        print(f"  扫描范围      : {result['lr_min']:.0e} → {result['lr_max']:.0e}")
        print(f"  总 step       : {result['total_steps']}")
        print(f"  下降最快处 lr : {result['steepest_descent_lr']:.2e}   ← 好 lr 的量级")
        print(f"  loss 最低处 lr: {result['loss_minimum_lr']:.2e}")
        print(f"  建议上界      : {result['suggested_lr_upper_bound']:.2e}   ← 别超过")
        print()
        print("  读法：取「下降最快处」的 1/3 ~ 1/10 作为实际 lr，留安全边际")
        (output_dir / "lr_range.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("结果已存 -> %s", output_dir / "lr_range.json")
        return 0

    # --- 其余三个模式都是配置列表 ---
    grid: dict[str, list[OptConfig]] = {
        "optimizers": OPTIMIZER_EXPERIMENTS,
        "warmup": WARMUP_EXPERIMENTS,
        "batch_size": BATCH_EXPERIMENTS,
    }
    experiments = grid[args.mode]
    if args.only:
        experiments = [e for e in experiments if args.only in e.name]
        if not experiments:
            logger.error("没有匹配 %r 的配置", args.only)
            return 2

    track = args.mode == "warmup"  # 只有 warmup 模式需要 step 级 lr
    results: list[dict[str, Any]] = []
    for index, config in enumerate(experiments, start=1):
        logger.info("=" * 70)
        logger.info("[%s/%s] %s —— %s", index, len(experiments), config.name, config.note)
        logger.info("=" * 70)
        row = runner.run(config, track=track)
        results.append(row)
        logger.info(
            "  -> train_loss=%.4f gap=%.4f ndcg最优=%.4f（最优轮 %s）",
            row["train_loss_last"],
            row["gap_loss"],
            row["valid_ndcg_best"],
            row["best_epoch"],
        )

    print_table(results)

    payload = {
        "mode": args.mode,
        "args": vars(args),
        "experiments": [asdict(e) for e in experiments],
        "results": results,
    }
    (output_dir / f"{args.mode}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("结果已存 -> %s", output_dir / f"{args.mode}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
