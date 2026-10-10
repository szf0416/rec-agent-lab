"""通用训练循环。

## 设计目标

**对"模型是什么"一无所知。** 只依赖 `TrainableModel` 协议，
所以 SASRec、DIN、双塔都能复用同一个 Trainer。

## 三个关键机制

1. **梯度累积**：显存不够时用小 batch 累积梯度，模拟大 batch。
   `optimizer.step()` 每 `accum_steps` 个 micro-batch 才执行一次。
   注意 loss 必须除以 `accum_steps`，否则等效学习率被放大 N 倍。
2. **warmup 与 scheduler 的执行顺序**：warmup 手动覆盖 lr，
   所以它必须在 `scheduler.step()` **之后**执行，否则会被 scheduler 覆盖，
   warmup 静默失效。
3. **checkpoint 带配置快照与 git commit**：只存 `state_dict` 的实验
   一个月后无法复现——你不知道它用什么超参跑出来的。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from recagent.train.callbacks import CallbackList, TrainHistory
from recagent.train.config import TrainConfig
from recagent.utils.logging import get_logger
from recagent.utils.seeding import seed_everything, seed_worker

logger = get_logger(__name__)


@runtime_checkable
class TrainableModel(Protocol):
    """训练循环视角下的模型接口。

    约定继承 `nn.Module`（这样能用 `train()`/`eval()`/`parameters()`），
    并额外实现 `forward` 与 `predict` 的语义。

    刻意**不强制**具体类型：测试时可以继承一个 20 行的极简 `nn.Module` 子类，
    不需要真的 SASRec。
    """

    def forward(self, batch: Any) -> torch.Tensor:
        """返回标量 loss。"""

    def predict(self, batch: Any) -> torch.Tensor:
        """返回 (B, num_candidates) 分数，供验证/评测使用。"""

    # 以下三个来自 nn.Module。显式声明是为了让类型检查器知道它们存在——
    # 只在文档里写"约定继承 nn.Module"是不够的，mypy 看不到这句约定，
    # 调用 self.model.parameters() 时会报 attr-defined。
    # 返回类型用 Any 以避免与 nn.Module 的具体签名冲突。
    def parameters(self) -> Any:
        """返回可训练参数迭代器。"""

    def state_dict(self) -> dict[str, Any]:
        """返回参数字典。"""

    def load_state_dict(self, state: dict[str, Any]) -> Any:
        """加载参数字典。"""


# --------------------------------------------------------------------------
# 学习率
# --------------------------------------------------------------------------


def warmup_lr(
    base_lr: float,
    step: int,
    warmup_steps: int,
    start_factor: float = 0.01,
) -> float:
    """线性预热：第 step 步应使用的基础学习率。

    为什么需要预热：训练初期参数是随机初始化的，梯度方向不稳定，
    直接上目标学习率容易跳进坏区域甚至发散。先用极小 lr 走几步
    等方向稳定，再线性升上去。这是 Transformer 训练的标准做法。
    """
    if warmup_steps <= 0 or step >= warmup_steps:
        return base_lr
    progress = step / max(warmup_steps, 1)
    return base_lr * (start_factor + (1.0 - start_factor) * progress)


def build_optimizer(model: TrainableModel, config: TrainConfig) -> torch.optim.Optimizer:
    """按配置构造优化器。

    AdamW 与 Adam + weight_decay 的区别（面试常问）：
    - Adam + L2：权重衰减被混进梯度里，与自适应学习率耦合，
      大梯度参数的实际衰减被削弱，正则效果不符合预期
    - AdamW：把权重衰减从梯度中解耦，直接作用于参数
    现代做法推荐 AdamW。
    """
    if not isinstance(model, nn.Module):
        raise TypeError(
            f"Trainer 要求模型继承 nn.Module（需要 train/eval/parameters），"
            f"收到 {type(model).__name__}"
        )
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("模型没有可训练参数，检查 requires_grad 设置")

    name = config.optimizer.lower()
    if name == "sgd":
        return torch.optim.SGD(
            params, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay
        )
    if name == "adam":
        return torch.optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    raise ValueError(f"未知优化器 {config.optimizer!r}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    *,
    steps_per_epoch: int,
) -> tuple[torch.optim.lr_scheduler.LRScheduler | None, str]:
    """按配置构造学习率调度器。

    Returns:
        (调度器或 None, interval)。interval 是 "step" 或 "epoch"，
        决定 Trainer 在哪里调用 `.step()`。

    注意：**本项目的 warmup 不用 PyTorch 的 `LambdaLR` 实现**。
    因为 warmup 与主调度器的组合在 LambdaLR 里很难写对（两个 lambda 相乘、
    且 epoch 级调度器在其中按 batch 调用会算错）。
    我们改为在 `Trainer._apply_lr()` 里逐 step 手动覆盖 warmup 阶段，
    warmup 结束后交还给 scheduler。逻辑直白，也更容易测。
    """
    schedule = config.scheduler
    if not schedule.enabled:
        return None, schedule.interval

    name = schedule.name.lower()
    total_steps = max(config.epochs * steps_per_epoch, 1)

    if name in {"step", "steplr"}:
        return (
            torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=schedule.step_size, gamma=schedule.gamma
            ),
            "epoch",
        )
    if name in {"multistep", "multi_step"}:
        if not schedule.milestones:
            raise ValueError("scheduler=multistep 需要提供 milestones 列表")
        return (
            torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=schedule.milestones, gamma=schedule.gamma
            ),
            "epoch",
        )
    if name == "cosine":
        return (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(config.epochs, 1),
                eta_min=config.lr * schedule.min_lr_ratio,
            ),
            "epoch",
        )
    if name in {"onecycle", "one_cycle"}:
        if schedule.interval != "step":
            raise ValueError("OneCycleLR 必须使用 interval='step'（配置校验应已拦截）")
        return (
            torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=config.lr, total_steps=total_steps, pct_start=0.3
            ),
            "step",
        )
    raise ValueError(f"未知 scheduler {config.scheduler.name!r}")


# --------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------


class Trainer:
    """通用训练循环。

    用法::

        trainer = Trainer(model, train_loader, valid_loader, config, callbacks)
        history = trainer.fit()
    """

    def __init__(
        self,
        model: TrainableModel,
        train_loader: DataLoader[Any],
        valid_loader: DataLoader[Any] | None,
        config: TrainConfig,
        callbacks: list[Any] | CallbackList | None = None,
    ) -> None:
        if len(train_loader) == 0:
            raise ValueError("train_loader 是空的，无法训练")
        if config.eval_every_n_epochs > 0 and valid_loader is None:
            logger.warning("未提供 valid_loader，将跳过验证，早停与选优无法生效")

        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.config = config
        self.callbacks = (
            callbacks if isinstance(callbacks, CallbackList) else CallbackList(callbacks)
        )
        self.history = TrainHistory()

        self.device = self._resolve_device(config.device)
        self.current_epoch = 0
        self.global_step = 0
        self._should_stop = False
        self._stop_reason = ""

        self._move_model_to_device()
        self.optimizer = build_optimizer(model, config)
        self.scheduler, self.scheduler_interval = build_scheduler(
            self.optimizer, config, steps_per_epoch=len(train_loader)
        )
        self._last_valid_metrics: dict[str, float] = {}
        self._best_metric: float | None = None

    # ------------------------------------------------------------------
    # 环境准备
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"配置要求 device={device}，但当前环境检测不到 CUDA。"
                f"请改用 device='cpu' 或 'auto'。"
            )
        return resolved

    def _move_model_to_device(self) -> None:
        self.model.to(self.device)  # type: ignore[attr-defined]

    def _maybe_rebuild_loader(self, loader: DataLoader[Any]) -> DataLoader[Any]:
        """按 num_workers 重建 DataLoader。

        为什么重建：`num_workers` 与 `worker_init_fn` 只能在构造时指定。
        外部传入的 loader 未必设对——而**可复现性要求 worker 进程也固定种子**，
        否则同一个 seed 跑两次结果仍可能不同。
        """
        if self.config.num_workers == 0:
            return loader
        return DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,  # type: ignore[arg-type]
            shuffle=isinstance(loader.sampler, torch.utils.data.RandomSampler),
            num_workers=self.config.num_workers,
            worker_init_fn=seed_worker,
            drop_last=getattr(loader, "drop_last", False),
        )

    # ------------------------------------------------------------------
    # 学习率
    # ------------------------------------------------------------------

    def _apply_step_lr(self) -> float:
        """推进一次 step 级学习率，返回实际值，并写回 optimizer。

        顺序至关重要：
        1. 先让 scheduler 按自己的规律更新（它会写 optimizer.param_groups）
        2. 再用手动 warmup 覆盖——**仅限 warmup 阶段内**

        如果反过来（先 warmup 再 scheduler.step），scheduler 会把 warmup
        刚设好的学习率覆盖掉，warmup 完全失效，而且不会报错。

        epoch 级调度器不走这里——它们在 fit() 的 epoch 结束时 `.step()`，
        调用次数才正确。
        """
        schedule = self.config.scheduler

        if self.scheduler is not None and self.scheduler_interval == "step":
            self.scheduler.step()

        lr = float(self.optimizer.param_groups[0]["lr"])

        if schedule.warmup_steps > 0 and self.global_step < schedule.warmup_steps:
            lr = warmup_lr(
                self.config.lr,
                self.global_step,
                schedule.warmup_steps,
                schedule.warmup_start_factor,
            )

        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------

    def train_epoch(self) -> dict[str, float]:
        """跑一个 epoch，返回训练指标。

        梯度累积逻辑：
        - 每个 micro-batch 都 backward
        - 每 accum_steps 个 micro-batch 才 step 一次（最后一个不完整的组也要 step）
        - loss 除以 accum_steps，保证等效学习率与真大 batch 一致
        """
        self.model.train()  # type: ignore[attr-defined]
        accum_steps = max(self.config.accum_steps, 1)
        total_loss = 0.0
        num_batches = 0
        grad_norms: list[float] = []

        self.optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(self.train_loader):
            moved = self._to_device(batch)
            loss = self.model.forward(moved)
            if not torch.is_tensor(loss) or loss.dim() != 0:
                raise TypeError(
                    f"模型 forward 必须返回标量 tensor，收到 {type(loss).__name__}，"
                    f"shape={getattr(loss, 'shape', None)}"
                )

            (loss / accum_steps).backward()

            is_group_end = (batch_index + 1) % accum_steps == 0
            is_last_batch = (batch_index + 1) == len(self.train_loader)
            total_loss += float(loss.detach().item())
            num_batches += 1

            if not (is_group_end or is_last_batch):
                continue

            if self.config.grad_clip is not None:
                # clip_grad_norm_ 返回裁剪前的总范数，顺手记录下来
                norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip,  # type: ignore[attr-defined]
                )
                grad_norms.append(float(norm))

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            lr = self._apply_step_lr()
            self.global_step += 1

            if (
                self.config.log_every_n_steps
                and self.global_step % self.config.log_every_n_steps == 0
            ):
                logger.info(
                    "epoch %s step %s/%s loss=%.4f lr=%.2e",
                    self.current_epoch,
                    batch_index + 1,
                    len(self.train_loader),
                    total_loss / max(num_batches, 1),
                    lr,
                )

        metrics: dict[str, float] = {
            "loss": total_loss / max(num_batches, 1),
            "num_steps": float(self.global_step),
            "lr": self._current_lr(),
        }
        if grad_norms:
            metrics["grad_norm"] = float(sum(grad_norms) / len(grad_norms))
            self.history.grad_norm_history.append(metrics["grad_norm"])
        return metrics

    # ------------------------------------------------------------------
    # 验证
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """在验证集上算指标。

        三个必须做对的地方：
        1. `@torch.no_grad()`——否则会构建计算图，显存持续增长直到 OOM
        2. `model.eval()`——切换 dropout / batchnorm 行为。忘记加会让验证结果
           每次都不一样，甚至比训练指标还差
        3. 只返回 loss；需要 `predict()` 的模型才额外算 top-k 指标
        """
        if self.valid_loader is None:
            return {}

        self.model.eval()  # type: ignore[attr-defined]
        total_loss = 0.0
        num_batches = 0
        score_chunks: list[np.ndarray] = []
        label_chunks: list[np.ndarray] = []

        has_predict = callable(getattr(self.model, "predict", None))

        for batch in self.valid_loader:
            moved = self._to_device(batch)
            loss = self.model.forward(moved)
            total_loss += float(loss.detach().item())
            num_batches += 1

            labels = self._extract_labels(moved)
            if not has_predict or labels is None:
                continue
            scores = self.model.predict(moved)
            score_chunks.append(scores.detach().float().cpu().numpy())
            label_chunks.append(labels.detach().cpu().numpy())

        metrics: dict[str, float] = {"loss": total_loss / max(num_batches, 1)}
        if score_chunks and label_chunks:
            metrics.update(
                self._ranking_metrics(np.concatenate(score_chunks), np.concatenate(label_chunks))
            )
        return metrics

    def _track_best(self, metrics: dict[str, float]) -> None:
        """记录并更新最优指标。

        为什么由 Trainer 负责而不是回调：
        `EarlyStopping` / `ModelCheckpoint` 都可能缺席（用户只想简单跑一跑），
        那样 `history.best_metric` 永远是 None，训练结束打印 `最优 loss=N/A`，
        让人误以为训练出了问题。

        **训练流程自己就该知道最优结果**，这是它的固有职责，
        不该依赖"用户注册了哪个回调"。
        """
        monitor = self.config.monitor
        if monitor not in metrics:
            return
        current = float(metrics[monitor])
        best = self._best_metric
        improved = best is None or (current > best if self.config.mode == "max" else current < best)
        if improved:
            self._best_metric = current
            self.history.best_metric = current
            self.history.best_epoch = self.current_epoch

    @staticmethod
    def _extract_labels(batch: Any) -> torch.Tensor | None:
        """从 batch 里取真实标签。

        统一约定多个候选 key，避免每个模型各写一套取标签逻辑。
        """
        if not isinstance(batch, dict):
            return None
        for key in ("labels", "label", "target", "targets"):
            value = batch.get(key)
            if torch.is_tensor(value):
                return value
        return None

    @staticmethod
    def _ranking_metrics(
        scores: np.ndarray, labels: np.ndarray, top_k: int = 10
    ) -> dict[str, float]:
        """通用 top-k 排序指标：HR / NDCG / MRR。

        评测协议：每行是一个用户的候选集，每行恰有一个正样本。
        这是"1 正 + N 负"的标准离线评测设置。

        **注意**：这是为了让训练循环能报告"可判断优劣"的数字而写的简版。
        S3 会实现完整的指标模块（含 Recall@K、覆盖率、多样性，
        以及与 RecBole 的交叉校验），届时替换本函数。
        """
        if scores.ndim != 2 or labels.ndim != 2 or scores.shape != labels.shape:
            return {}
        k = min(top_k, scores.shape[1])
        if k < 1:
            return {}

        # 每行取分数最高的 k 个（argpartition 是 O(n)，比全排序快）
        topk_idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        # 组内再按分数精确排序：take_along_axis 需要与 topk_idx 同形的索引
        topk_sorted = np.take_along_axis(
            topk_idx, np.argsort(-np.take_along_axis(scores, topk_idx, axis=1), axis=1), axis=1
        )

        hit = np.zeros(scores.shape[0], dtype=bool)
        ndcg = np.zeros(scores.shape[0], dtype=float)
        mrr = np.zeros(scores.shape[0], dtype=float)

        for row in range(scores.shape[0]):
            positive = np.flatnonzero(labels[row] > 0)
            if positive.size == 0:
                continue
            matches = np.flatnonzero(topk_sorted[row] == int(positive[0]))
            if matches.size == 0:
                continue
            rank = int(matches[0])  # 0-based
            hit[row] = True
            ndcg[row] = 1.0 / np.log2(rank + 2)
            mrr[row] = 1.0 / (rank + 1)

        # 分母用"有正样本的用户数"，而不是总行数——否则缺正样本的行会拉低指标
        num_users = max(int(np.sum(np.any(labels > 0, axis=1))), 1)
        return {
            f"hr@{k}": float(hit.sum()) / num_users,
            f"ndcg@{k}": float(ndcg.sum()) / num_users,
            f"mrr@{k}": float(mrr.sum()) / num_users,
        }

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def fit(self, *, resume_from: str | Path | None = None) -> TrainHistory:
        """执行完整训练，返回训练历史。"""
        seed_everything(self.config.seed)

        train_loader = self._maybe_rebuild_loader(self.train_loader)
        valid_loader = (
            self._maybe_rebuild_loader(self.valid_loader) if self.valid_loader is not None else None
        )
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        start_epoch = 0
        if resume_from is not None:
            metadata = self.load_checkpoint(Path(resume_from))
            start_epoch = int(metadata.get("epoch", -1)) + 1
            logger.info("从 %s 恢复训练，起始 epoch=%s", resume_from, start_epoch)

        self.callbacks.on_train_start(self, self.history)
        start_time = time.time()

        try:
            for epoch in range(start_epoch, self.config.epochs):
                self.current_epoch = epoch
                self.callbacks.on_epoch_start(self, self.history)

                train_metrics = self.train_epoch()
                self.history.train_metrics.append(train_metrics)
                self.history.lr_history.append(train_metrics["lr"])

                should_validate = (
                    self.valid_loader is not None
                    and self.config.eval_every_n_epochs > 0
                    and (epoch + 1) % self.config.eval_every_n_epochs == 0
                )
                if should_validate:
                    valid_metrics = self.validate()
                    self.history.valid_metrics.append(valid_metrics)
                    self._last_valid_metrics = valid_metrics
                    self._track_best(valid_metrics)
                    logger.info(
                        "epoch %s | train_loss=%.4f | valid=%s",
                        epoch,
                        train_metrics["loss"],
                        {key: round(value, 4) for key, value in valid_metrics.items()},
                    )
                else:
                    logger.info("epoch %s | train_loss=%.4f", epoch, train_metrics["loss"])

                # epoch 级调度器在此推进：每个 epoch 恰好一次
                if self.scheduler is not None and self.scheduler_interval == "epoch":
                    self.scheduler.step()

                self.callbacks.on_epoch_end(self, self.history)

                if self._should_stop:
                    logger.info("训练提前结束：%s", self._stop_reason)
                    break
        finally:
            # 保证即使中途抛异常，MLflow run 也会被关闭
            self.callbacks.on_train_end(self, self.history)

        elapsed = time.time() - start_time
        best = self.history.best_metric
        logger.info(
            "训练结束 | %s 个 epoch | 耗时 %.1f 秒 | 最优 %s=%s @ epoch %s",
            self.history.num_epochs,
            elapsed,
            self.config.monitor,
            f"{best:.6f}" if best is not None else "N/A",
            self.history.best_epoch,
        )
        return self.history

    # ------------------------------------------------------------------
    # 回调接口
    # ------------------------------------------------------------------

    def stop_training(self, reason: str = "") -> None:
        self._should_stop = True
        self._stop_reason = reason

    def save_checkpoint(self, path: Path, metrics: dict[str, float]) -> Path:
        """保存完整训练状态。

        **必须包含配置快照和 git commit**，否则一个月后无法复现实验。
        只存 `model.state_dict()` 是最常见的错误——
        你拿到一个 best.pt，却不知道它是用什么超参、哪份代码训出来的。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state: dict[str, Any] = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "metrics": dict(metrics),
            "best_metric": self.history.best_metric,
            "best_epoch": self.history.best_epoch,
            "config": self.config.to_dict(),
            "git_commit": _git_commit(),
            "torch_version": torch.__version__,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        torch.save(state, path)
        logger.debug("已保存 checkpoint: %s", path)
        return path

    def load_checkpoint(self, path: Path) -> dict[str, Any]:
        """读取 checkpoint 并恢复全部状态。

        `weights_only=False` 是必需的：checkpoint 里有 config dict 和 str，
        而 PyTorch 2.6+ 默认 `weights_only=True` 只允许张量与基础类型。
        我们只加载自己训练产生的文件，风险可控。
        """
        payload: dict[str, Any] = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model"])
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        if "scheduler" in payload and self.scheduler is not None:
            self.scheduler.load_state_dict(payload["scheduler"])
        if "global_step" in payload:
            self.global_step = int(payload["global_step"])
        best = payload.get("best_metric")
        if best is not None:
            self._best_metric = float(best)
            self.history.best_metric = float(best)
            self.history.best_epoch = int(payload.get("best_epoch", -1))
        return payload

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    def _to_device(self, batch: Any) -> Any:
        """把 batch 里的张量搬到目标设备，保持原有结构。

        递归处理 dict / list / tuple：推荐模型的 batch 通常是
        `{"user_id": Tensor, "seq": Tensor, "labels": Tensor}` 这类嵌套结构。
        """
        if torch.is_tensor(batch):
            return batch.to(self.device, non_blocking=True)
        if isinstance(batch, dict):
            return {key: self._to_device(value) for key, value in batch.items()}
        if isinstance(batch, (list, tuple)):
            return type(batch)(self._to_device(item) for item in batch)
        return batch


def _git_commit() -> str:
    """取当前 git commit hash。取不到就返回 unknown——不能因此让训练失败。"""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"
