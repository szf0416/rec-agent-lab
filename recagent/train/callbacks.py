"""训练回调。

## 为什么用回调，而不是把 early stop 写进 Trainer.fit()

这是本周最值得想清楚的设计问题。四个理由：

1. **开闭原则**：想加"记录梯度范数""画 lr 曲线"时，不改 Trainer 一行代码。
2. **可测试**：回调是独立对象，可以脱离真实训练做单元测试。
   如果你把 early stop 写在 `fit()` 里，要测它就得跑一次真的训练。
3. **可组合**：不同实验用不同回调组合，靠配置驱动（S3 的消融实验需要这个）。
4. **关注点分离**：Trainer 只管"训练流程"，策略属于回调。

**判断抽象是否合格的方法**：接 MLflow 时如果发现需要改 Trainer，说明抽象漏了。

## 依赖方向

本模块**不导入** `recagent.train.loop`。回调需要访问 trainer，
通过下面这个 Protocol 声明它实际需要的最小接口。

这样做的意义：`loop.py` 依赖 `callbacks.py`（它调用回调），
反过来不依赖。单向依赖，没有循环导入，也便于单独测试回调。
"""

from __future__ import annotations

import contextlib
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from recagent.utils.logging import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------
# Trainer 的最小接口
# --------------------------------------------------------------------------


@runtime_checkable
class TrainerLike(Protocol):
    """回调视角下的 Trainer。

    只声明回调真正会用到的成员。**故意不声明** batch 循环相关的内部方法——
    声明得越少，回调与 Trainer 的耦合越松，将来换实现越容易。
    """

    current_epoch: int
    model: Any
    config: Any

    def stop_training(self, reason: str = "") -> None:
        """请求提前结束训练。由 Trainer 在下一个检查点响应。"""

    def save_checkpoint(self, path: Path, metrics: dict[str, float]) -> Path:
        """保存完整训练状态（模型 + 优化器 + 调度器 + 配置快照 + 进度）。"""

    def load_checkpoint(self, path: Path) -> dict[str, Any]:
        """读取 checkpoint，返回其中的 metadata。"""


# --------------------------------------------------------------------------
# 训练历史
# --------------------------------------------------------------------------


@dataclass
class TrainHistory:
    """逐 epoch 的训练记录。

    存在的意义：没有它，训练完只能看到最后一个 epoch 的数字，
    无法判断"哪一轮最优""什么时候开始过拟合"。
    这两个判断是做消融和写报告的基础。
    """

    train_metrics: list[dict[str, float]] = field(default_factory=list)
    valid_metrics: list[dict[str, float]] = field(default_factory=list)
    lr_history: list[float] = field(default_factory=list)
    grad_norm_history: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_metric: float | None = None
    stopped_early: bool = False
    stop_reason: str = ""

    # --- 便捷查询 ---

    @property
    def num_epochs(self) -> int:
        return len(self.train_metrics)

    def metric_series(self, name: str, *, source: str = "valid") -> list[float]:
        """取某个指标随 epoch 变化的序列。

        Args:
            name: 指标名，如 "ndcg@10"、"loss"。
            source: "valid" 取验证集记录，"train" 取训练集记录。
        """
        records = self.valid_metrics if source == "valid" else self.train_metrics
        return [r[name] for r in records if name in r]

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_epochs": self.num_epochs,
            "train_metrics": self.train_metrics,
            "valid_metrics": self.valid_metrics,
            "lr_history": self.lr_history,
            "grad_norm_history": self.grad_norm_history,
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "stopped_early": self.stopped_early,
            "stop_reason": self.stop_reason,
        }


# --------------------------------------------------------------------------
# 回调协议
# --------------------------------------------------------------------------


class Callback(Protocol):
    """训练过程中的钩子。

    所有方法都有默认实现的语义（不实现就是不做事），
    所以回调只需要覆盖自己关心的钩子。
    """

    def on_train_start(self, trainer: TrainerLike, history: TrainHistory) -> None: ...
    def on_train_end(self, trainer: TrainerLike, history: TrainHistory) -> None: ...
    def on_epoch_start(self, trainer: TrainerLike, history: TrainHistory) -> None: ...
    def on_epoch_end(self, trainer: TrainerLike, history: TrainHistory) -> None: ...


class BaseCallback:
    """提供空实现的基类。

    为什么需要它：`Protocol` 只描述接口，不给默认实现。
    如果回调必须显式实现全部 4 个方法，写一个只关心 `on_epoch_end` 的回调
    就得写 3 个 `pass`。用基类继承更省事。
    """

    def on_train_start(self, trainer: TrainerLike, history: TrainHistory) -> None:
        del trainer, history  # 未使用，由子类覆盖

    def on_train_end(self, trainer: TrainerLike, history: TrainHistory) -> None:
        del trainer, history

    def on_epoch_start(self, trainer: TrainerLike, history: TrainHistory) -> None:
        del trainer, history

    def on_epoch_end(self, trainer: TrainerLike, history: TrainHistory) -> None:
        del trainer, history


# --------------------------------------------------------------------------
# CallbackList：把一组回调当成一个回调
# --------------------------------------------------------------------------


class CallbackList:
    """按顺序分发钩子调用。

    设计要点：**注册顺序决定调用顺序**。
    例如 ModelCheckpoint 必须在 EarlyStopping 之前注册，
    否则早停触发时已经恢复了最优权重，checkpoint 保存的就不是
    "本轮实际训练得到"的权重。
    """

    def __init__(self, callbacks: list[Any] | None = None) -> None:
        self.callbacks: list[Any] = list(callbacks or [])

    def append(self, callback: Any) -> None:
        self.callbacks.append(callback)

    def __len__(self) -> int:
        return len(self.callbacks)

    def __iter__(self) -> Any:
        return iter(self.callbacks)

    def _invoke(self, hook: str, trainer: TrainerLike, history: TrainHistory) -> None:
        for callback in self.callbacks:
            method = getattr(callback, hook, None)
            if method is not None:
                method(trainer, history)

    def on_train_start(self, trainer: TrainerLike, history: TrainHistory) -> None:
        self._invoke("on_train_start", trainer, history)

    def on_train_end(self, trainer: TrainerLike, history: TrainHistory) -> None:
        self._invoke("on_train_end", trainer, history)

    def on_epoch_start(self, trainer: TrainerLike, history: TrainHistory) -> None:
        self._invoke("on_epoch_start", trainer, history)

    def on_epoch_end(self, trainer: TrainerLike, history: TrainHistory) -> None:
        self._invoke("on_epoch_end", trainer, history)


# --------------------------------------------------------------------------
# EarlyStopping
# --------------------------------------------------------------------------


class EarlyStopping(BaseCallback):
    """验证指标不再提升时停止训练，并恢复最优权重。

    ## 为什么必须恢复最优权重（任务卡的「坑 2」）

    早期停止的典型场景：

        epoch 10: NDCG 0.310   ← 最优
        epoch 11: NDCG 0.300
        epoch 12: NDCG 0.295   ← patience 用完，停止

    若只"停止"不"恢复"，训练结束时模型停在 epoch 12（0.295），
    **你把最好的那个模型丢掉了，而且不会报错**。

    本实现默认 `restore_best=True`，在停止前把权重恢复成最优 epoch 的状态。

    ## 与 ModelCheckpoint 的分工

    两者都会追踪最优指标，但职责不同：
    - ModelCheckpoint 负责**持久化**（写磁盘，可跨进程恢复）
    - EarlyStopping 负责**停止决策**（可只存内存，训练结束即释放）

    因此 restore_best 用内存快照而不是要求磁盘上有 checkpoint——
    这样即使用户没配 ModelCheckpoint，早停也能正确工作。
    """

    def __init__(
        self,
        monitor: str = "loss",
        patience: int = 20,
        mode: str = "min",
        min_delta: float = 0.0,
        restore_best: bool = True,
    ) -> None:
        if mode not in {"max", "min"}:
            raise ValueError(f"mode 必须是 max/min，收到 {mode!r}")
        if patience < 1:
            raise ValueError(f"patience 必须 >= 1，收到 {patience}")

        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best = restore_best

        self.best: float | None = None
        self.best_epoch: int = -1
        self.wait: int = 0
        self.should_stop: bool = False
        # 内存中的最优权重快照
        self._best_state: dict[str, Any] | None = None

    # -- 比较逻辑 --

    def _is_improvement(self, current: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "max":
            return current > self.best + self.min_delta
        return current < self.best - self.min_delta

    def on_epoch_end(self, trainer: TrainerLike, history: TrainHistory) -> None:
        metrics = history.valid_metrics[-1] if history.valid_metrics else {}
        if self.monitor not in metrics:
            # 明确报警而不是沉默。配错 monitor 名会让早停永远不触发，
            # 训练跑到 max_epochs 才发现，白白浪费几小时。
            warnings.warn(
                f"EarlyStopping 监控的指标 {self.monitor!r} 不在本轮指标 "
                f"{sorted(metrics)} 中，早停不会生效。请检查 monitor 配置。",
                stacklevel=2,
            )
            return

        current = float(metrics[self.monitor])

        if self._is_improvement(current):
            self.best = current
            self.best_epoch = trainer.current_epoch
            self.wait = 0
            # 注意：**不在这里写 history.best_metric**。
            # Trainer._track_best() 已经负责记录最优指标，两处都写会互相覆盖，
            # 出现"history 说 best 在 epoch 10，早停说在 epoch 12"这种矛盾。
            # 回调的职责是"做决策"（何时停止），不是"记录统计"。
            if self.restore_best:
                self._best_state = _snapshot_state_dict(trainer.model)
            return

        self.wait += 1
        if self.wait < self.patience:
            return

        # --- 触发早停 ---
        self.should_stop = True
        history.stopped_early = True
        history.stop_reason = (
            f"{self.monitor} 连续 {self.patience} 轮未提升"
            f"（最优 {self.best:.6f} @ epoch {self.best_epoch}）"
        )

        if self.restore_best and self._best_state is not None:
            trainer.model.load_state_dict(self._best_state)
            logger.info(
                "早停：已恢复 epoch %s 的最优权重（%s=%.6f）",
                self.best_epoch,
                self.monitor,
                self.best,
            )

        trainer.stop_training(history.stop_reason)


def _snapshot_state_dict(model: Any) -> dict[str, Any]:
    """做一份权重深拷贝。

    为什么不用 `copy.deepcopy(model)`：那会连优化器状态、计算图引用一起复制，
    既慢又可能带出不需要的东西。我们只要参数本身。
    """
    import copy

    state = model.state_dict()
    return {key: copy.deepcopy(value) for key, value in state.items()}


# --------------------------------------------------------------------------
# ModelCheckpoint
# --------------------------------------------------------------------------


class ModelCheckpoint(BaseCallback):
    """保存 top-k 最优 checkpoint，并维护 `last.pt` 用于断点续训。

    命名约定：
    - `epoch{N}_...pt`：按监控指标选出
    - `last.pt`：每个 epoch 覆盖，用于中断后恢复

    为什么两者都要：
    - 只要 best：中断后无法续训（不知道训练进度、优化器动量状态）
    - 只要 last：可能保存的是过拟合后的模型
    """

    def __init__(
        self,
        directory: str | Path,
        monitor: str = "loss",
        mode: str = "min",
        save_top_k: int = 1,
        save_last: bool = True,
        filename_template: str = "epoch{epoch:03d}_{metric}{value:.4f}.pt",
    ) -> None:
        if mode not in {"max", "min"}:
            raise ValueError(f"mode 必须是 max/min，收到 {mode!r}")
        if save_top_k < 1:
            raise ValueError(f"save_top_k 必须 >= 1，收到 {save_top_k}")

        self.directory = Path(directory)
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.filename_template = filename_template

        self.best: float | None = None
        self.best_path: Path | None = None
        # (指标值, 路径) 列表，按优劣排序
        self.saved: list[tuple[float, Path]] = []

    def _is_better(self, a: float, b: float) -> bool:
        return a > b if self.mode == "max" else a < b

    def on_epoch_end(self, trainer: TrainerLike, history: TrainHistory) -> None:
        metrics = history.valid_metrics[-1] if history.valid_metrics else {}
        if self.monitor not in metrics:
            warnings.warn(
                f"ModelCheckpoint 监控的指标 {self.monitor!r} 不在本轮指标 "
                f"{sorted(metrics)} 中，不会保存最优模型（只保存 last）。",
                stacklevel=2,
            )
            if self.save_last:
                self._save(trainer, metrics, "last.pt")
            return

        current = float(metrics[self.monitor])

        # 第一个 epoch 无条件保存，保证至少有产物
        best = self.best
        if best is None or self._is_better(current, best):
            self.best = current
            name = self.filename_template.format(
                epoch=trainer.current_epoch,
                metric=self.monitor.replace("/", "_").replace("@", "at"),
                value=current,
            )
            path = self._save(trainer, metrics, name)
            self.best_path = path
            self.saved.append((current, path))

            # 维护 top-k：超出就删掉最差的那个
            if len(self.saved) > self.save_top_k:
                self.saved.sort(key=lambda pair: pair[0], reverse=(self.mode == "max"))
                _, worst_path = self.saved.pop()
                if worst_path.exists():
                    worst_path.unlink()
                    logger.debug(
                        "已删除被挤出 top-%s 的 checkpoint: %s",
                        self.save_top_k,
                        worst_path.name,
                    )

        if self.save_last:
            self._save(trainer, metrics, "last.pt")

    def _save(self, trainer: TrainerLike, metrics: dict[str, float], filename: str) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        return trainer.save_checkpoint(self.directory / filename, metrics)


# --------------------------------------------------------------------------
# MLflowLogger
# --------------------------------------------------------------------------


class MLflowLogger(BaseCallback):
    """把超参与逐 epoch 指标记录到 MLflow。

    设计要点：**延迟导入 mlflow**。
    没装 mlflow 时应该给出明确的降级警告，而不是让整个训练失败——
    实验追踪是"有用的附加物"，不该成为训练的必要条件。

    默认使用 `start_run` + `end_run` 手动配对（而非 `with` 块），
    因为训练过程跨越多个回调方法，`with` 的作用域无法覆盖。
    """

    def __init__(
        self,
        experiment_name: str = "recagent",
        run_name: str | None = None,
        tracking_uri: str | None = None,
        log_artifacts: bool = False,
    ) -> None:
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.log_artifacts = log_artifacts
        self._mlflow: Any = None
        self._active = False

    def on_train_start(self, trainer: TrainerLike, history: TrainHistory) -> None:
        del history
        try:
            import mlflow
        except ImportError:
            warnings.warn(
                "未安装 mlflow，实验追踪已跳过。安装后可获得完整实验记录：pip install mlflow",
                stacklevel=2,
            )
            return

        self._mlflow = mlflow
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=self.run_name)
        self._active = True

        config = trainer.config
        payload = config.to_dict() if hasattr(config, "to_dict") else {}
        # 记录超参。嵌套的 scheduler 需要展平，MLflow 只接受扁平标量。
        flat: dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat[f"{key}.{sub_key}"] = sub_value
            else:
                flat[key] = value
        mlflow.log_params(
            {k: v for k, v in flat.items() if isinstance(v, (str, int, float, bool)) or v is None}
        )

    def on_epoch_end(self, trainer: TrainerLike, history: TrainHistory) -> None:
        if not self._active or self._mlflow is None:
            return

        step = trainer.current_epoch
        if history.train_metrics:
            self._mlflow.log_metrics(
                {f"train/{k}": float(v) for k, v in history.train_metrics[-1].items()},
                step=step,
            )
        if history.valid_metrics:
            self._mlflow.log_metrics(
                {f"valid/{k}": float(v) for k, v in history.valid_metrics[-1].items()},
                step=step,
            )
        if history.lr_history:
            self._mlflow.log_metric("lr", float(history.lr_history[-1]), step=step)

    def on_train_end(self, trainer: TrainerLike, history: TrainHistory) -> None:
        del trainer
        if not self._active or self._mlflow is None:
            return

        if history.best_metric is not None:
            self._mlflow.log_metric("best_metric", float(history.best_metric))
        self._mlflow.log_metric("best_epoch", float(history.best_epoch))

        if self.log_artifacts:
            summary = json.dumps(history.to_dict(), ensure_ascii=False, indent=2)
            self._mlflow.log_text(summary, "train_history.json")

        self._mlflow.end_run()
        self._active = False

    def __del__(self) -> None:
        """确保进程异常退出时 run 被关闭，避免 MLflow 留下悬挂的 run。

        用 `suppress` 而不是 `try/except/pass`：解释器关闭阶段的异常
        无法处理也无法上报，静默忽略是唯一选择，`suppress` 让意图更明确。
        """
        if self._active and self._mlflow is not None:
            with contextlib.suppress(Exception):
                self._mlflow.end_run(status="FAILED")
