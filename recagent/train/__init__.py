"""训练子包：通用训练循环、配置与回调。

设计要点见 `docs/tasks/S1-W2-train-loop.md`：

- `Trainer` 对模型一无所知，只依赖 `TrainableModel` 协议
- 策略（早停 / checkpoint / 追踪）通过回调插拔，不改 Trainer
- 所有超参走 `TrainConfig`，配置校验在训练开始前完成
"""

from recagent.train.callbacks import (
    BaseCallback,
    Callback,
    CallbackList,
    EarlyStopping,
    MLflowLogger,
    ModelCheckpoint,
    TrainerLike,
    TrainHistory,
)
from recagent.train.config import (
    PER_STEP_SCHEDULER_NAMES,
    SchedulerConfig,
    TrainConfig,
)
from recagent.train.loop import (
    TrainableModel,
    Trainer,
    build_optimizer,
    build_scheduler,
    warmup_lr,
)

__all__ = [
    "PER_STEP_SCHEDULER_NAMES",
    "BaseCallback",
    "Callback",
    "CallbackList",
    "EarlyStopping",
    "MLflowLogger",
    "ModelCheckpoint",
    "SchedulerConfig",
    "TrainConfig",
    "TrainHistory",
    "TrainableModel",
    "Trainer",
    "TrainerLike",
    "build_optimizer",
    "build_scheduler",
    "warmup_lr",
]
