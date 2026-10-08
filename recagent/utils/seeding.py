"""随机种子与确定性。

为什么单独成模块：这个项目的所有实验都要求「同 seed 跑两次结果一致」，
散落各处的 `np.random.seed()` 迟早会漏掉一处，导致结果不可复现。
统一入口，并在训练启动时显式调用。
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np

_DEFAULT_SEED = 42


def seed_everything(seed: int = _DEFAULT_SEED, *, deterministic: bool = True) -> int:
    """固定所有随机源。

    Args:
        seed: 随机种子。
        deterministic: 是否强制 cuDNN 确定性算法。
            开启后结果可复现，但可能变慢；追求速度时可关闭。

    Returns:
        实际使用的 seed，便于记录到实验日志。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # torch 非必需（例如只跑数据层测试时），故延迟导入
    try:
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed


def seed_worker(worker_id: int) -> None:
    """DataLoader worker 的初始化函数，保证多进程加载也可复现。"""
    del worker_id  # 未使用，但签名由 torch 约定
    worker_seed = _DEFAULT_SEED
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def numpy_rng(seed: int = _DEFAULT_SEED) -> np.random.Generator:
    """返回独立的 numpy 随机数生成器。

    推荐优先使用本函数而不是 `np.random.*` 全局函数：
    独立生成器不会互相干扰，便于并行实验。
    """
    return np.random.default_rng(seed)


def describe_rng_state() -> dict[str, Any]:
    """当前随机状态摘要，写入实验记录用于排查不可复现问题。"""
    state: dict[str, Any] = {
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "random_state": random.getstate()[1][:3],
    }
    try:
        import torch
    except ImportError:
        return state
    state["torch_seed"] = torch.initial_seed()
    if torch.cuda.is_available():
        state["cuda_seed"] = torch.cuda.initial_seed()
    return state
