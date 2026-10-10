"""训练配置。

设计原则：**训练需要什么参数，全部在这里声明**，不允许散落在代码里。

为什么用 dataclass 而不是 dict：
1. 类型检查能在写代码时发现拼写错误（`cfg.lr_` 会立刻报错，`cfg["lr_"]` 只在运行时炸）
2. IDE 能自动补全，不用反复翻 yaml
3. 有默认值，想调什么就改什么

为什么单独成模块而不写在 loop.py 里：
配置是"契约"，训练循环是"实现"。契约放前面，实现可以换（比如以后换
transformers.Trainer），合约不变。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# scheduler 的步进时机。用错会让学习率曲线完全不是你想要的，且不会报错。
# 详见 docs/tasks/S1-W2-train-loop.md 的「坑 1」。
SchedulerInterval = Literal["step", "epoch"]

# 已知的 per-step 调度器。用于在 interval 配置错误时主动报警——
# PyTorch 没有标准接口能自动判断，只能用类型名做启发式检查。
# 不完美，但"已知易错点主动报警"比沉默好得多。
PER_STEP_SCHEDULER_NAMES = frozenset(
    {
        "OneCycleLR",
        "CosineAnnealingWarmRestarts",
        "CyclicLR",
    }
)


@dataclass
class SchedulerConfig:
    """学习率调度配置。

    Attributes:
        name: 调度器名。none 表示不使用。
        interval: step 表示每个 batch 后更新；epoch 表示每个 epoch 结束更新。
        warmup_steps: 线性预热步数。前 N 步把 lr 从 ~0 升到目标值。
            训练初期梯度方向不稳定，直接上大 lr 容易发散，预热能显著稳定训练。
        warmup_start_factor: 预热起点相对目标 lr 的比例。
        min_lr_ratio: 余弦衰减的下限（相对目标 lr 的比例）。
        step_size / gamma / milestones: 供 StepLR / MultiStepLR 使用。
    """

    name: str = "none"
    interval: SchedulerInterval = "epoch"
    warmup_steps: int = 0
    warmup_start_factor: float = 0.01
    min_lr_ratio: float = 0.0
    step_size: int = 10
    gamma: float = 0.1
    milestones: list[int] = field(default_factory=list)

    @property
    def enabled(self) -> bool:
        return self.name.lower() != "none"

    @property
    def needs_total_steps(self) -> bool:
        """这些调度器在构造时必须知道总步数，否则无法建立曲线。"""
        return self.name in {"onecycle", "cosine"}


@dataclass
class TrainConfig:
    """一次训练的完整配置。

    Attributes:
        epochs: 最大训练轮数（早停可能提前结束）。
        batch_size: 批大小。
        lr: 学习率。
        weight_decay: 权重衰减（L2 正则）。注意 AdamW 是解耦实现，
            与 Adam + weight_decay 行为不同。
        optimizer: sgd | adam | adamw。
        momentum: 仅 sgd 使用。
        grad_clip: 梯度裁剪的 max_norm。None 表示不裁剪。
            推荐模型的 embedding 梯度容易爆炸，建议开启。
        early_stop_patience: 验证指标连续多少轮没提升就停止。None 表示不早停。
        early_stop_min_delta: 指标提升需要超过这个幅度才算"有提升"，
            避免被浮点噪声触发。
        early_stop_restore_best: 早停时是否恢复最优权重。
            **强烈建议 True**——否则你会丢掉最好的模型（见任务卡「坑 2」）。
        eval_every_n_epochs: 每隔多少轮做一次验证。1 表示每轮都验证。
        num_workers: DataLoader 工作进程数。Windows 下建议 0。
        amp: 是否启用自动混合精度。6GB 显存下可省显存，但前期建议关闭以稳定优先。
        device: cuda | cpu | auto。auto 表示有 GPU 就用。
        seed: 随机种子。
        log_every_n_steps: 每多少 step 打印一次训练日志。0 表示不打印。
        checkpoint_dir: checkpoint 输出目录。
        monitor: 早停与选优所依据的指标名。
        mode: max 表示指标越大越好，min 表示越小越好。
    """

    # --- 训练规模 ---
    epochs: int = 200
    batch_size: int = 128

    # --- 优化 ---
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adam"
    momentum: float = 0.9
    grad_clip: float | None = 5.0
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    accum_steps: int = 1
    """梯度累积步数。>1 时每 N 个 micro-batch 才执行一次 optimizer.step()，
    用显存换等效 batch size。实现里 loss 会被除以 N 以保持等效学习率不变。"""

    # --- 早停与选优 ---
    early_stop_patience: int | None = 20
    early_stop_min_delta: float = 0.0
    early_stop_restore_best: bool = True
    monitor: str = "loss"
    mode: Literal["max", "min"] = "min"

    # --- 运行 ---
    eval_every_n_epochs: int = 1
    num_workers: int = 0
    amp: bool = False
    device: str = "auto"
    seed: int = 42
    log_every_n_steps: int = 50
    checkpoint_dir: str = "outputs/checkpoints"

    # ------------------------------------------------------------------
    # 校验：把"能提前发现的错误"全部挡在训练开始之前
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """配置自检。

        为什么值得写这么多校验：训练一次可能几小时，
        参数写错却要等到第 3 个 epoch 才暴露，代价太大。
        宁可启动时直接报错。
        """
        if self.epochs < 1:
            raise ValueError(f"epochs 必须 >= 1，收到 {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size 必须 >= 1，收到 {self.batch_size}")
        if self.lr <= 0:
            raise ValueError(f"lr 必须 > 0，收到 {self.lr}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay 必须 >= 0，收到 {self.weight_decay}")
        if self.grad_clip is not None and self.grad_clip <= 0:
            raise ValueError(f"grad_clip 必须 > 0 或 None，收到 {self.grad_clip}")
        if self.early_stop_patience is not None and self.early_stop_patience < 1:
            raise ValueError(
                f"early_stop_patience 必须 >= 1 或 None，收到 {self.early_stop_patience}"
            )
        if self.eval_every_n_epochs < 1:
            raise ValueError(f"eval_every_n_epochs 必须 >= 1，收到 {self.eval_every_n_epochs}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers 必须 >= 0，收到 {self.num_workers}")
        if self.accum_steps < 1:
            raise ValueError(f"accum_steps 必须 >= 1，收到 {self.accum_steps}")
        if self.optimizer.lower() not in {"sgd", "adam", "adamw"}:
            raise ValueError(f"optimizer 必须是 sgd/adam/adamw 之一，收到 {self.optimizer!r}")
        if self.mode not in {"max", "min"}:
            raise ValueError(f"mode 必须是 max/min，收到 {self.mode!r}")
        if self.scheduler.warmup_steps < 0:
            raise ValueError(f"warmup_steps 必须 >= 0，收到 {self.scheduler.warmup_steps}")
        self._validate_scheduler_interval()

    def _validate_scheduler_interval(self) -> None:
        """检查 scheduler 类型与 interval 是否匹配。

        这是本项目最容易犯的错之一：OneCycleLR 是 per-step 调度器，
        若按 epoch 调用，本该 1000 步走完的曲线被压缩到几十步，
        学习率曲线完全不是你想要的，**而且不会报错**。

        PyTorch 没有标准接口能查询调度器的步进单位，只能按类型名做启发式判断。
        我们选择"主动报警"而不是沉默通过。
        """
        if not self.scheduler.enabled:
            return
        if self.scheduler.name in PER_STEP_SCHEDULER_NAMES and self.scheduler.interval == "epoch":
            raise ValueError(
                f"scheduler={self.scheduler.name} 是 per-step 调度器，"
                f"但 interval 配置为 'epoch'。这会让学习率曲线被压缩到 1/n_batches，"
                f"且不会报错。请把 interval 改为 'step'。"
            )

    def to_dict(self) -> dict[str, Any]:
        """转成可序列化的 dict，用于写进 checkpoint 和实验追踪。

        为什么需要它：checkpoint 里必须带配置快照。
        否则一个月后你看到某个 `best.pt`，根本不知道它是用什么超参训出来的。
        """
        payload: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, SchedulerConfig):
                payload[key] = {
                    "name": value.name,
                    "interval": value.interval,
                    "warmup_steps": value.warmup_steps,
                    "warmup_start_factor": value.warmup_start_factor,
                    "min_lr_ratio": value.min_lr_ratio,
                    "step_size": value.step_size,
                    "gamma": value.gamma,
                    "milestones": list(value.milestones),
                }
            else:
                payload[key] = value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainConfig:
        """从 dict 构造，兼容 yaml 里 scheduler 用嵌套 dict 的写法。

        注意 scheduler 是唯一的嵌套字段，要单独转成 SchedulerConfig，
        否则 dataclass 会把 dict 原样塞进去，访问 `.interval` 时才炸。
        """
        data = dict(payload)
        scheduler = data.get("scheduler")
        if isinstance(scheduler, dict):
            # 过滤掉 SchedulerConfig 不认识的键，避免 yaml 里多写一个字段就报错。
            # 但要报出被忽略的键——静默丢弃配置是最危险的行为之一。
            known = set(SchedulerConfig.__dataclass_fields__)
            unknown = set(scheduler) - known
            if unknown:
                raise ValueError(
                    f"scheduler 配置里有无法识别的字段: {sorted(unknown)}。"
                    f"可用字段: {sorted(known)}"
                )
            data["scheduler"] = SchedulerConfig(**scheduler)
        return cls(**data)
