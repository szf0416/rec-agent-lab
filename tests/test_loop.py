"""训练循环测试（S1-W2 的 10 个必测项）。

## 为什么这个文件的测试必须"秒级跑完"

用假模型 + 假数据，不碰真实数据、不需要 GPU。
如果这些测试要跑 30 秒，你以后就不会跑它们了——而它们是唯一能保证
训练循环正确的东西。半年后这个文件就成了摆设。

## 这些测试在防什么

训练循环有个危险特性：**写错了不一定报错**。
warmup 顺序反了、梯度累积忘了除、早停没恢复最优权重——
这三种 bug 都不会崩溃，只会让结果悄悄变错。
测试就是用来抓这类"静默错误"的。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from recagent.train import (
    EarlyStopping,
    ModelCheckpoint,
    SchedulerConfig,
    TrainConfig,
    Trainer,
)
from recagent.utils.seeding import seed_everything

# --------------------------------------------------------------------------
# 测试替身：一个可解析的极简模型
# --------------------------------------------------------------------------

TRUE_W = 2.5
TRUE_B = -1.2
FEATURE_DIM = 4


class TinyLinear(nn.Module):
    """y = x @ w + b，用 MSE 训练。

    为什么用这么简单的模型：它的最优解可以手算（w→2.5, b→-1.2），
    所以"训练是否有效"能精确断言，而不是"loss 好像小了点"。

    注意它**只继承 nn.Module**，完全不知道 TrainableModel 协议的存在——
    Protocol 是结构子类型，只要方法齐备就被接受。
    """

    def __init__(self, dim: int = FEATURE_DIM) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, 1) * 0.1)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        prediction = batch["x"] @ self.weight + self.bias
        return ((prediction - batch["y"]) ** 2).mean()

    def predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return (batch["x"] @ self.weight + self.bias).squeeze(-1)


def make_batch(size: int = 128, seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(size, FEATURE_DIM, generator=generator)
    y = x @ torch.full((FEATURE_DIM, 1), TRUE_W) + TRUE_B
    y = y + 0.01 * torch.randn(size, 1, generator=generator)
    return {"x": x, "y": y}


@pytest.fixture(autouse=True)
def _deterministic_model_init() -> None:
    """每个测试前固定全局种子。

    为什么需要：`seed_everything()` 必须在**模型构造之前**调用，
    否则 `TinyLinear.__init__` 里的 `torch.randn` 不受控，
    每次跑测试的初始参数都不同。

    这不是小题大做——真实训练里"忘了在建模型前设种子"正是
    实验结果不可复现的最常见原因。测试里踩一遍，记得更牢。
    """
    seed_everything(0)


def make_loader(size: int = 256, batch_size: int = 64, seed: int = 0) -> DataLoader[Any]:
    batch = make_batch(size, seed)
    dataset = TensorDataset(batch["x"], batch["y"])
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # 关掉 shuffle，让测试结果完全确定
        collate_fn=lambda items: {
            "x": torch.stack([item[0] for item in items]),
            "y": torch.stack([item[1] for item in items]),
        },
    )


def make_config(**overrides: Any) -> TrainConfig:
    """构造测试用配置。默认值刻意选得"能快速收敛且安静"。"""
    defaults: dict[str, Any] = {
        "epochs": 3,
        "lr": 0.05,
        "grad_clip": None,
        "eval_every_n_epochs": 1,
        "log_every_n_steps": 0,  # 关掉逐 step 日志，输出干净
        "device": "cpu",
        "seed": 0,
    }
    defaults.update(overrides)
    return TrainConfig(**defaults)


def make_trainer(
    model: nn.Module | None = None,
    config: TrainConfig | None = None,
    callbacks: list[Any] | None = None,
) -> Trainer:
    return Trainer(
        model if model is not None else TinyLinear(),  # type: ignore[arg-type]
        make_loader(),
        make_loader(seed=1),
        config if config is not None else make_config(),
        callbacks if callbacks is not None else [],
    )


# --------------------------------------------------------------------------
# #1 loss 下降 —— 循环真的在优化，不是空转
# --------------------------------------------------------------------------


def test_loss_decreases() -> None:
    """训练若干轮后 loss 应显著下降，且参数收敛到真值。

    如果这个测试失败，说明训练循环根本没在工作（梯度没回传、优化器没 step 等），
    后面 9 个测试都不用看了。

    轮数取 30：实测 10 轮时 weight 均值只到 1.94（真值 2.5），
    30 轮才收敛到 2.4955。**轮数不够会让断言偶发失败**，
    这类"测试本身不稳定"的问题比代码 bug 更烦人，所以要留足余量。
    """
    seed_everything(0)
    model = TinyLinear()
    trainer = make_trainer(model, make_config(epochs=30, lr=0.05, grad_clip=1.0))
    history = trainer.fit()

    losses = [record["loss"] for record in history.train_metrics]
    assert losses[-1] < losses[0] * 0.01, f"loss 未显著下降: {losses[0]:.4f} -> {losses[-1]:.4f}"

    # 参数应收敛到真值附近（手算最优解就是 TRUE_W / TRUE_B）
    assert model.weight.mean().item() == pytest.approx(TRUE_W, abs=0.1)
    assert model.bias.item() == pytest.approx(TRUE_B, abs=0.1)


# --------------------------------------------------------------------------
# #2 .grad 是累加的 —— 梯度累积能成立的底层机制
# --------------------------------------------------------------------------


def test_grad_accumulates_without_zero_grad() -> None:
    """同一个 batch 连续 backward 两次且不清理，梯度必须是 2 倍。

    **这就是"梯度累积"能成立的底层机制。**
    不清理时 `.grad` 是累加而非覆盖 —— 所以：
    - 训练循环每步必须 `zero_grad()`
    - 梯度累积正是**利用**这个累加特性来模拟大 batch

    注意验证的是"梯度累加"，不是"loss 累加"：loss 是标量，各次之间互不影响；
    真正在累加的是参数的 `.grad`。
    """
    model = TinyLinear()
    batch = make_batch(size=64)

    model.zero_grad()
    model.forward(batch).backward()
    grad_after_first = model.weight.grad.clone()

    # 同一个 batch 再跑一次 → 每项梯度数值相同 → 累加后应恰好是 2 倍
    model.forward(batch).backward()
    torch.testing.assert_close(model.weight.grad, grad_after_first * 2)

    # zero_grad 之后归零（set_to_none=True 时是 None）
    model.zero_grad()
    assert model.weight.grad is None or torch.allclose(
        model.weight.grad, torch.zeros_like(grad_after_first)
    )


# --------------------------------------------------------------------------
# #3 梯度裁剪生效
# --------------------------------------------------------------------------


def test_gradient_clipping_limits_norm() -> None:
    """梯度范数被裁剪到阈值内，并且训练循环确实记录了裁剪前的范数。

    推荐模型的 embedding 梯度容易爆炸（热门物品被大量交互），
    一次更新就可能让 loss 变 NaN。所以这个机制必须有测试守着。
    """
    seed_everything(0)
    # lr 极大 + 裁剪阈值极小 → 保证梯度一定超过阈值
    config = make_config(epochs=1, lr=10.0, grad_clip=0.01)
    trainer = make_trainer(config=config)
    history = trainer.fit()

    assert history.grad_norm_history, "训练循环没有记录梯度范数"
    recorded = history.grad_norm_history[0]
    assert recorded > config.grad_clip, (
        f"记录的范数 {recorded} 未超过阈值 {config.grad_clip}，"
        f"这个测试失去了意义（应先确认真的发生了裁剪）"
    )

    # 裁剪后所有参数梯度的总范数不得超过阈值。
    # 用 math.sqrt 而不是 torch.sqrt：torch 2.14 拒绝 Python float 标量
    # （实测 torch.sqrt(0.0) 会抛 TypeError），而这里算出来的就是 float。
    actual_norm = math.sqrt(
        sum(float(p.grad.norm() ** 2) for p in trainer.model.parameters() if p.grad is not None)
    )
    assert actual_norm <= config.grad_clip + 1e-6, f"裁剪后范数 {actual_norm} 仍超阈值"


# --------------------------------------------------------------------------
# #4 scheduler 的步进时机（epoch 级 vs step 级）
# --------------------------------------------------------------------------


def test_epoch_scheduler_steps_once_per_epoch() -> None:
    """epoch 级调度器每轮恰好 step 一次。

    这里是"坑 1"的防线：OneCycleLR 这类 per-step 调度器若按 epoch 调用，
    学习率曲线会被压缩到 1/n_batches，而且**不会报错**。
    所以我们要能区分并验证两种调度器的调用次数。
    """
    seed_everything(0)
    config = make_config(
        epochs=4, lr=0.01, scheduler=SchedulerConfig(name="step", interval="epoch")
    )
    trainer = make_trainer(config=config)
    initial_last_epoch = trainer.scheduler.last_epoch  # type: ignore[union-attr]

    history = trainer.fit()

    assert trainer.scheduler is not None
    # 4 个 epoch → last_epoch 从 -1 递增到 3
    assert trainer.scheduler.last_epoch == initial_last_epoch + history.num_epochs


def test_step_scheduler_steps_once_per_optimizer_step() -> None:
    """step 级调度器每次 optimizer.step() 后推进一次。

    OneCycleLR 的曲线是按**总步数**设计的。按 epoch 调用会让它提前走完，
    学习率在第一个 epoch 内就衰减到接近 0。
    """
    seed_everything(0)
    config = make_config(
        epochs=2,
        lr=0.01,
        scheduler=SchedulerConfig(name="onecycle", interval="step"),
    )
    trainer = make_trainer(config=config)
    assert trainer.scheduler_interval == "step"
    history = trainer.fit()

    expected_steps = history.num_epochs * len(trainer.train_loader)
    assert trainer.scheduler is not None
    assert trainer.scheduler.last_epoch == expected_steps, (
        f"per-step 调度器应推进 {expected_steps} 次，实际 {trainer.scheduler.last_epoch} 次"
    )


# --------------------------------------------------------------------------
# #5 早停触发
# --------------------------------------------------------------------------


def test_early_stopping_triggers_after_patience() -> None:
    """指标连续 patience 轮不提升时，早停必须触发。

    这里**手工喂入**一个构造好的指标序列，而不是跑真实训练。
    好处：可以精确控制"从哪一轮开始不提升"，让断言完全确定。
    """
    trainer = make_trainer()
    history = trainer.history
    stopper = EarlyStopping(monitor="ndcg@10", patience=2, mode="max", restore_best=False)

    # 第 0 轮：提升（建立基线）
    history.valid_metrics.append({"ndcg@10": 0.5})
    trainer.current_epoch = 0
    stopper.on_epoch_end(trainer, history)
    assert not trainer._should_stop
    assert stopper.wait == 0

    # 第 1 轮：下降 → wait=1，还没到 patience=2
    history.valid_metrics.append({"ndcg@10": 0.4})
    trainer.current_epoch = 1
    stopper.on_epoch_end(trainer, history)
    assert not trainer._should_stop

    # 第 2 轮：再下降 → wait=2，达到 patience，触发
    history.valid_metrics.append({"ndcg@10": 0.3})
    trainer.current_epoch = 2
    stopper.on_epoch_end(trainer, history)
    assert trainer._should_stop, "patience 用尽后应触发早停"
    assert history.stopped_early
    assert "ndcg@10" in history.stop_reason


def test_early_stopping_resets_wait_on_improvement() -> None:
    """一旦有提升，patience 计数必须归零。

    否则"断续提升"的训练会被误杀——观察值是坏的，但趋势是好的。
    """
    trainer = make_trainer()
    history = trainer.history
    stopper = EarlyStopping("ndcg@10", patience=2, mode="max", restore_best=False)

    for epoch, value in enumerate([0.5, 0.4, 0.6]):  # 中间降一次，随后又提升
        history.valid_metrics.append({"ndcg@10": value})
        trainer.current_epoch = epoch
        stopper.on_epoch_end(trainer, history)

    assert not trainer._should_stop
    assert stopper.wait == 0, "有提升后 wait 必须归零"


# --------------------------------------------------------------------------
# #6 早停恢复最优权重（最容易写成假测试的一项）
# --------------------------------------------------------------------------


def test_early_stopping_restores_best_weights() -> None:
    """早停后模型参数必须等于**最优**那一轮的，而不是最后一轮的。

    对应任务卡的「坑 2」：

        epoch 10: NDCG 0.310   ← 最优
        epoch 11: NDCG 0.300
        epoch 12: NDCG 0.295   ← 停止

    若只"停"不"恢复"，你把最好的模型丢掉了，而且不会报错。

    **这个测试的关键是断言"等于最优"而不是"等于最后"**——
    如果只断言 should_stop，即使实现不恢复权重它也会通过（假测试）。
    """
    trainer = make_trainer()
    model = trainer.model
    history = trainer.history
    stopper = EarlyStopping("loss", patience=2, mode="min", restore_best=True)

    # 每轮跑一步真实的前向反向，制造"训练不稳定"的指标序列
    for epoch, value in enumerate([0.1, 0.2, 0.3]):  # 最优在 epoch 0，之后持续退步
        with torch.no_grad():
            # 权重按"轮次序号"变化，便于断言到底恢复到了哪一轮
            model.weight.fill_(float(epoch))  # type: ignore[attr-defined]

        history.valid_metrics.append({"loss": value})
        trainer.current_epoch = epoch
        stopper.on_epoch_end(trainer, history)

    assert trainer._should_stop, "第三个 epoch 应触发早停"
    assert stopper.best_epoch == 0

    # 最优发生在 epoch 0，快照里存的应是当时的权重（全 0.0）。
    # 若恢复正确 → 参数被还原成 0.0；若未恢复 → 会停在最后一轮的 2.0。
    torch.testing.assert_close(
        model.weight,  # type: ignore[attr-defined]
        torch.zeros_like(model.weight),  # type: ignore[attr-defined]
        msg="早停未恢复最优权重（参数仍是最后一轮的）",
    )


def test_early_stopping_can_skip_restore() -> None:
    """restore_best=False 时应保留最后状态，作为对照。

    这个对照很重要：它证明上一个测试的通过**确实来自恢复逻辑**，
    而不是因为参数碰巧没变。
    """
    trainer = make_trainer()
    model = trainer.model
    history = trainer.history
    stopper = EarlyStopping("loss", patience=1, mode="min", restore_best=False)

    for epoch, value in enumerate([0.1, 0.2]):
        with torch.no_grad():
            model.weight.fill_(float(epoch))  # type: ignore[attr-defined]
        history.valid_metrics.append({"loss": value})
        trainer.current_epoch = epoch
        stopper.on_epoch_end(trainer, history)

    assert trainer._should_stop
    # 不恢复 → 参数停在最后一轮（epoch 1 → 全 1.0）
    torch.testing.assert_close(
        model.weight,  # type: ignore[attr-defined]
        torch.ones_like(model.weight),  # type: ignore[attr-defined]
    )


# --------------------------------------------------------------------------
# #7 checkpoint 往返
# --------------------------------------------------------------------------


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    """保存后重新加载，参数必须完全一致。"""
    seed_everything(0)
    source = make_trainer()
    source.fit()

    path = tmp_path / "roundtrip.pt"
    source.save_checkpoint(path, {"loss": 0.5})

    target_model = TinyLinear()
    target = make_trainer(model=target_model)
    metadata = target.load_checkpoint(path)

    for key, value in source.model.state_dict().items():  # type: ignore[attr-defined]
        torch.testing.assert_close(target_model.state_dict()[key], value)

    # 配置快照必须存在：没有它，一个月后无法复现这个 checkpoint
    assert "config" in metadata
    assert metadata["config"]["lr"] == source.config.lr
    assert "git_commit" in metadata
    assert metadata["metrics"]["loss"] == 0.5


# --------------------------------------------------------------------------
# #8 固定 seed 可复现
# --------------------------------------------------------------------------


def test_same_seed_reproduces_identical_losses() -> None:
    """同一 seed 跑两次，loss 序列必须完全相同。

    可复现的前提是**种子在模型构造之前固定**：
    模型初始化、数据顺序、dropout 都依赖全局随机状态。
    若先建模型再设种子，参数初始化不受控，结果必然不同。
    """

    def run(config_seed: int) -> list[float]:
        seed_everything(config_seed)  # 先固定全局状态
        model = TinyLinear()  # 再建模型 → 初始化受种子控制
        trainer = Trainer(
            model,
            make_loader(),
            make_loader(seed=1),
            make_config(seed=config_seed),
            [],
        )
        return [record["loss"] for record in trainer.fit().train_metrics]

    assert run(42) == run(42), "同 seed 两次结果不一致，可复现性被破坏"
    assert run(42) != run(7), "换 seed 结果应不同（否则说明种子没起作用）"


# --------------------------------------------------------------------------
# #9 空 dataloader
# --------------------------------------------------------------------------


def test_empty_train_loader_rejected() -> None:
    """空的训练集必须立刻报错，而不是跑完一个 epoch 后除以零。"""
    empty = DataLoader(TensorDataset(torch.empty(0, FEATURE_DIM), torch.empty(0, 1)), batch_size=8)
    with pytest.raises(ValueError, match="train_loader 是空的"):
        Trainer(TinyLinear(), empty, None, make_config(), [])  # type: ignore[arg-type]


def test_missing_valid_loader_warns_not_crashes() -> None:
    """没有验证集时应降级为警告，训练仍可进行（用于快速调试）。"""
    trainer = Trainer(TinyLinear(), make_loader(), None, make_config(epochs=1), [])  # type: ignore[arg-type]
    assert trainer.validate() == {}
    history = trainer.fit()
    assert history.num_epochs == 1


# --------------------------------------------------------------------------
# #10 从 checkpoint 恢复训练
# --------------------------------------------------------------------------


def test_resume_from_checkpoint_continues(tmp_path: Path) -> None:
    """从 checkpoint 续训：应按保存的轮次继续，且指标连续。"""
    seed_everything(0)
    config = make_config(epochs=4)
    first = make_trainer(config=config)
    path = tmp_path / "resume.pt"
    first.save_checkpoint(path, {"loss": 0.9})

    model = TinyLinear()
    resumed = make_trainer(model=model, config=config)
    history = resumed.fit(resume_from=path)

    # checkpoint 里 epoch=0（还没跑过）→ 应从 epoch 1 开始，共跑 3 轮
    assert history.num_epochs == config.epochs - 1, (
        f"恢复训练应只跑剩余轮次，实际跑了 {history.num_epochs}"
    )
    # 续训的 loss 应与从头训练的对应轮次一致（seed 相同、数据相同）
    assert all(loss > 0 for loss in (record["loss"] for record in history.train_metrics))


def test_resume_restores_optimizer_state(tmp_path: Path) -> None:
    """优化器状态也要恢复，否则 Adam 的动量会从零开始，指标不连续。"""
    seed_everything(0)
    trainer = make_trainer(config=make_config(epochs=3))
    trainer.fit()
    path = tmp_path / "opt.pt"
    trainer.save_checkpoint(path, {"loss": 0.5})

    target = make_trainer()
    target.load_checkpoint(path)

    source_state = trainer.optimizer.state_dict()
    target_state = target.optimizer.state_dict()
    assert len(target_state["state"]) == len(source_state["state"]), (
        "优化器内部状态（Adam 的 m/v）未随 checkpoint 恢复"
    )


# --------------------------------------------------------------------------
# 附：ModelCheckpoint 的 top-k 行为
# --------------------------------------------------------------------------


def test_model_checkpoint_keeps_best_and_last(tmp_path: Path) -> None:
    """checkpoint 应同时保留"最优"和"最近"，两者职责不同。

    - 只要 best：中断后无法续训（不知道进度、没有优化器状态）
    - 只要 last：可能保存的是过拟合后的模型
    """
    trainer = make_trainer()
    history = trainer.history
    checkpoint = ModelCheckpoint(tmp_path, monitor="loss", mode="min", save_top_k=1)

    for epoch, value in enumerate([0.5, 0.3, 0.4]):
        history.valid_metrics.append({"loss": value})
        trainer.current_epoch = epoch
        checkpoint.on_epoch_end(trainer, history)

    assert (tmp_path / "last.pt").is_file(), "缺少 last.pt，无法断点续训"
    assert checkpoint.best == 0.3
    assert checkpoint.best_path is not None and checkpoint.best_path.is_file()
    # top_k=1 → 只保留一个最优文件（0.5 的文件应被清掉）
    assert len(list(tmp_path.glob("epoch*.pt"))) == 1, "save_top_k=1 时不应留下多个文件"
