"""训练配置测试。

除了校验逻辑，本文件还包含一条**配置同步守卫**：
确保 `configs/model/train_default.yaml` 的字段与 `TrainConfig` 完全对应。

为什么需要它：yaml 里漏写一个字段**不会报错**——dataclass 会用默认值。
于是你改了配置文件、以为生效了，实际训练用的是默认值。
这类"静默不生效"是最难发现的问题之一，必须用测试锁住。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from recagent.train.config import (
    PER_STEP_SCHEDULER_NAMES,
    SchedulerConfig,
    TrainConfig,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_YAML = PROJECT_ROOT / "configs" / "model" / "train_default.yaml"


# --------------------------------------------------------------------------
# 配置同步守卫
# --------------------------------------------------------------------------


def load_train_section() -> dict:
    """读取 train_default.yaml 的 train 段。"""
    with DEFAULT_TRAIN_YAML.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    assert "train" in payload, f"{DEFAULT_TRAIN_YAML.name} 缺少顶层 train 段"
    return dict(payload["train"])


def test_yaml_covers_every_train_config_field() -> None:
    """TrainConfig 的每个字段都必须在 yaml 里出现。

    漏字段的后果是静默的：改配置文件不生效，实际用默认值。
    """
    yaml_fields = set(load_train_section())
    dataclass_fields = set(TrainConfig.__dataclass_fields__)

    missing = dataclass_fields - yaml_fields
    assert not missing, f"train_default.yaml 缺少以下字段，改它们将不会生效：{sorted(missing)}"


def test_yaml_has_no_unknown_fields() -> None:
    """yaml 里不能有多余字段。

    多余字段更危险：写 `learning_rate: 0.01` 而真实字段叫 `lr`，
    训练照常跑，只是你设的学习率从未被使用。
    """
    yaml_fields = set(load_train_section())
    dataclass_fields = set(TrainConfig.__dataclass_fields__)

    extra = yaml_fields - dataclass_fields
    assert not extra, f"train_default.yaml 有无法识别的字段（会被忽略）：{sorted(extra)}"


def test_yaml_scheduler_section_matches_dataclass() -> None:
    """scheduler 子配置也要完全对应。"""
    yaml_fields = set(load_train_section()["scheduler"])
    dataclass_fields = set(SchedulerConfig.__dataclass_fields__)
    assert yaml_fields == dataclass_fields, (
        f"scheduler 配置不一致。缺少: {sorted(dataclass_fields - yaml_fields)}，"
        f"多余: {sorted(yaml_fields - dataclass_fields)}"
    )


def test_default_yaml_is_loadable() -> None:
    """默认配置必须能真的构造出 TrainConfig（含类型转换）。"""
    config = TrainConfig.from_dict(load_train_section())
    assert config.optimizer in {"sgd", "adam", "adamw"}
    assert config.epochs >= 1
    assert config.scheduler.interval in {"step", "epoch"}


# --------------------------------------------------------------------------
# 校验逻辑：把错误挡在训练开始之前
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"epochs": 0},
        {"epochs": -1},
        {"batch_size": 0},
        {"lr": 0.0},
        {"lr": -0.001},
        {"weight_decay": -0.1},
        {"grad_clip": 0},
        {"grad_clip": -1.0},
        {"accum_steps": 0},
        {"eval_every_n_epochs": 0},
        {"num_workers": -1},
        {"optimizer": "rmsprop"},
        {"mode": "up"},
        {"early_stop_patience": 0},
    ],
)
def test_invalid_config_rejected(overrides: dict) -> None:
    """非法配置必须在构造时立刻报错，而不是训练到一半才暴露。"""
    with pytest.raises(ValueError):
        TrainConfig(**overrides)


def test_grad_clip_none_is_allowed() -> None:
    """grad_clip=None 表示不裁剪，是合法配置。"""
    config = TrainConfig(grad_clip=None)
    assert config.grad_clip is None


def test_early_stop_patience_none_is_allowed() -> None:
    """不早停也是合法的（用于跑满 epochs 观察完整曲线）。"""
    assert TrainConfig(early_stop_patience=None).early_stop_patience is None


# --------------------------------------------------------------------------
# scheduler interval 校验（最容易静默出错的地方）
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(PER_STEP_SCHEDULER_NAMES))
def test_per_step_scheduler_rejects_epoch_interval(name: str) -> None:
    """per-step 调度器配 interval='epoch' 必须报错。

    这个错误不会自己暴露——学习率曲线被压缩到 1/n_batches，
    训练照跑，只是结果不对。所以宁可启动时直接失败。
    """
    with pytest.raises(ValueError, match="per-step"):
        TrainConfig(scheduler=SchedulerConfig(name=name, interval="epoch"))


@pytest.mark.parametrize("name", sorted(PER_STEP_SCHEDULER_NAMES))
def test_per_step_scheduler_accepts_step_interval(name: str) -> None:
    config = TrainConfig(scheduler=SchedulerConfig(name=name, interval="step"))
    assert config.scheduler.interval == "step"


def test_disabled_scheduler_skips_interval_check() -> None:
    """name=none 时不该被 interval 校验影响。"""
    config = TrainConfig(scheduler=SchedulerConfig(name="none", interval="epoch"))
    assert not config.scheduler.enabled


def test_negative_warmup_rejected() -> None:
    with pytest.raises(ValueError, match="warmup_steps"):
        TrainConfig(scheduler=SchedulerConfig(warmup_steps=-1))


# --------------------------------------------------------------------------
# 序列化：checkpoint 里要能存下完整配置
# --------------------------------------------------------------------------


def test_to_dict_roundtrip() -> None:
    """to_dict → from_dict 必须无损，否则 checkpoint 里的配置快照不可信。"""
    original = TrainConfig(
        epochs=7,
        lr=0.002,
        accum_steps=3,
        scheduler=SchedulerConfig(name="cosine", interval="epoch", warmup_steps=100),
    )
    restored = TrainConfig.from_dict(original.to_dict())

    assert restored.epochs == original.epochs
    assert restored.lr == original.lr
    assert restored.scheduler.name == original.scheduler.name
    assert restored.scheduler.warmup_steps == original.scheduler.warmup_steps
    assert restored.to_dict() == original.to_dict()


def test_to_dict_flattens_scheduler() -> None:
    """scheduler 必须展开成 dict，否则 mlflow.log_params 会拒收嵌套值。"""
    payload = TrainConfig(scheduler=SchedulerConfig(name="cosine")).to_dict()
    assert isinstance(payload["scheduler"], dict)
    assert payload["scheduler"]["name"] == "cosine"


def test_from_dict_rejects_unknown_scheduler_field() -> None:
    """scheduler 里的未知字段要报错，而不是静默丢弃。

    静默丢弃配置是最危险的行为——你以为改了，其实没生效。
    """
    with pytest.raises(ValueError, match="无法识别"):
        TrainConfig.from_dict({"scheduler": {"name": "cosine", "typo_field": 1}})
