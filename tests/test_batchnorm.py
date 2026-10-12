"""BatchNorm 行为验证（进 CI）。

对应 S1 验收标准第 3 条：

    能回答：BatchNorm 训练/推理为什么不同？

## 本文件守护的三个关键行为

1. **训练用 batch 统计、推理用滑动平均** —— 两者输出必须不同
2. **推理必须能用 batch=1** —— 而 batch 统计在这种情况下会失效（方差为 0）
3. **滑动平均存无偏方差、归一化用有偏方差** —— 一个容易写错的实现细节

## 为什么在测试里重新实现而不是 import from_scratch

与 `tests/test_gradient_check.py` 一致：`from_scratch/` 不是包、
文件名以数字开头，无法正常 import。测试保持自包含，不依赖教学脚本。

教学脚本（含更详细的演示）见 `from_scratch/04_batchnorm.py`。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


class ManualBatchNorm1d:
    """纯 NumPy 的 BatchNorm1d（教学脚本里的实现在此复刻一份）。

    两个容易写错的细节：
    - 归一化用**有偏**方差（ddof=0），滑动平均存**无偏**方差（ddof=1）
    - 训练要先算 batch 统计并更新 running，再归一化
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(num_features, dtype=np.float64)
        self.beta = np.zeros(num_features, dtype=np.float64)
        self.running_mean = np.zeros(num_features, dtype=np.float64)
        self.running_var = np.ones(num_features, dtype=np.float64)
        self.training = True

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.training:
            mean = x.mean(axis=0)
            var_biased = x.var(axis=0)
            batch_size = x.shape[0]
            var_unbiased = var_biased * batch_size / max(batch_size - 1, 1)

            self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (
                1.0 - self.momentum
            ) * self.running_var + self.momentum * var_unbiased
            normalized = (x - mean) / np.sqrt(var_biased + self.eps)
        else:
            normalized = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

        return self.gamma * normalized + self.beta


def make_input(rows: int = 16, cols: int = 4, seed: int = 0) -> np.ndarray:
    """非 0 均值、非 1 方差的数据 —— 这样"归一化"才有可观察的效果。"""
    return np.random.default_rng(seed).normal(5.0, 3.0, (rows, cols))


def make_torch_bn(cols: int = 4) -> torch.nn.BatchNorm1d:
    """构造与手写实现同参数的 PyTorch BN。

    `.double()` 是必需的：BN 的 weight/bias 默认 float32，
    与 float64 输入混用会报 "mixed dtype (CPU)"。这个坑在写教学脚本时踩过一次。
    """
    return torch.nn.BatchNorm1d(cols, eps=1e-5, momentum=0.1).double()


# --------------------------------------------------------------------------
# 1. 与 PyTorch 对齐
# --------------------------------------------------------------------------


def test_train_mode_matches_pytorch() -> None:
    """训练模式：手写实现与 PyTorch 的输出、滑动平均统计量都要一致。"""
    x = make_input()

    manual = ManualBatchNorm1d(4)
    manual.training = True
    out_manual = manual(x)

    torch_bn = make_torch_bn(4)
    torch_bn.train()
    out_torch = torch_bn(torch.tensor(x, dtype=torch.float64)).detach().numpy()

    assert float(np.max(np.abs(out_manual - out_torch))) < 1e-10

    # 归一化后每列均值≈0、标准差≈1
    assert np.allclose(out_manual.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(out_manual.std(axis=0), 1.0, atol=1e-6)


def test_eval_mode_matches_pytorch() -> None:
    """推理模式：用滑动平均统计量，也要与 PyTorch 一致。"""
    x = make_input()

    manual = ManualBatchNorm1d(4)
    manual.training = True
    manual(x)  # 累积一次统计量
    manual.training = False
    out_manual = manual(x)

    torch_bn = make_torch_bn(4)
    torch_bn.train()
    torch_bn(torch.tensor(x, dtype=torch.float64))
    torch_bn.eval()
    out_torch = torch_bn(torch.tensor(x, dtype=torch.float64)).detach().numpy()

    assert float(np.max(np.abs(out_manual - out_torch))) < 1e-10


def test_running_var_uses_unbiased_estimate() -> None:
    """滑动平均里的方差必须是**无偏**估计（除以 N-1），而归一化用有偏（除以 N）。

    这是个容易写错的实现细节：只差一个 N/(N-1) 因子，
    在小 batch 上会造成可观察的偏差。

    相对差距推导（bi=有偏, un=无偏）：
        bi = SS/N,  un = SS/(N-1)
        |bi - un| / un = |1/N - 1/(N-1)| / (1/(N-1)) = 1/N
    所以 N=16 时是 6.25%（不是我一开始算的 1/15 = 6.67%）。
    """
    x = make_input(rows=16)

    manual = ManualBatchNorm1d(4)
    manual.training = True
    manual(x)

    torch_bn = make_torch_bn(4)
    torch_bn.train()
    torch_bn(torch.tensor(x, dtype=torch.float64))

    assert np.allclose(manual.running_var, torch_bn.running_var.detach().numpy(), atol=1e-12), (
        "running_var 与 PyTorch 不一致，很可能是漏了「无偏」这个细节"
    )

    # 直接用有偏方差更新会差多少？（说明这个细节不是无关紧要）
    biased = x.var(axis=0)
    unbiased = biased * x.shape[0] / (x.shape[0] - 1)
    relative_gap = float(np.mean(np.abs(biased - unbiased) / unbiased))
    assert relative_gap == pytest.approx(1.0 / x.shape[0], abs=1e-6), (
        "N=16 时两者相差 1/N = 6.25%"
    )


# --------------------------------------------------------------------------
# 2. 核心：训练与推理行为不同
# --------------------------------------------------------------------------


def test_train_and_eval_produce_different_outputs() -> None:
    """同一个输入、同一组参数，训练与推理的输出必须不同。

    这是「BN 训练/推理行为不同」最直接的证据：
    - 训练输出被归一化到均值 ≈ 0
    - 推理输出用的是 running_mean（初始为 0，而输入均值是 5），所以均值不是 0

    实测最大差异约 6.22，远大于任何数值误差。
    """
    x = make_input()

    bn = ManualBatchNorm1d(4)
    bn.training = True
    out_train = bn(x)
    bn.training = False
    out_eval = bn(x)

    assert np.allclose(out_train.mean(axis=0), 0.0, atol=1e-10)
    assert not np.allclose(out_eval.mean(axis=0), 0.0, atol=1e-3), (
        "推理输出均值若为 0，说明误用了 batch 统计"
    )

    max_diff = float(np.max(np.abs(out_train - out_eval)))
    assert max_diff > 1.0, f"两种模式差异过小（{max_diff:.4f}），实现可能有问题"


# --------------------------------------------------------------------------
# 3. 为什么推理必须用滑动平均
# --------------------------------------------------------------------------


def test_batch_statistics_break_on_single_sample() -> None:
    """只有 1 个样本时，batch 方差恒为 0，归一化会丢光信息。

    这是"推理必须用滑动平均"最硬的理由 —— 线上请求是逐条来的。
    """
    x_single = np.array([[5.0, 3.0, -1.0, 2.0]])

    batch_var = x_single.var(axis=0)
    assert np.allclose(batch_var, 0.0), "单样本的方差必然为 0"

    # 强行用 batch 统计归一化
    normalized = (x_single - x_single.mean(axis=0)) / np.sqrt(batch_var + 1e-5)
    assert np.allclose(normalized, 0.0), "结果为全 0，信息完全丢失"


def test_eval_mode_works_with_single_sample() -> None:
    """而用滑动平均统计量时，单样本推理能得到有意义的输出。"""
    bn = ManualBatchNorm1d(4)
    bn.training = True
    bn(make_input(rows=32, seed=1))  # 用历史 batch 累积统计量
    bn.training = False

    out = bn(np.array([[5.0, 3.0, -1.0, 2.0]]))
    assert out.shape == (1, 4)
    assert not np.allclose(out, 0.0), "滑动平均下输出不应退化为全 0"
    assert np.all(np.isfinite(out))


# --------------------------------------------------------------------------
# 4. 训练时输出依赖 batch 组成（特性也是风险）
# --------------------------------------------------------------------------


def test_train_output_depends_on_other_samples_in_batch() -> None:
    """训练时，同一个样本的输出会随同 batch 的其它样本变化。

    这是 BN 的**特性**（batch 统计带来的正则效果），
    也是它的**风险**（小 batch 或分布不均时统计不稳定）。

    同时也解释了另一个常见现象：忘记 `model.eval()` 会让验证结果不稳定，
    因为验证时还在用 batch 统计。
    """
    rng = np.random.default_rng(2)
    sample = rng.normal(5.0, 3.0, (1, 4))

    bn = ManualBatchNorm1d(4)
    bn.training = True

    batch_a = np.concatenate([sample, rng.normal(0.0, 1.0, (15, 4))])
    batch_b = np.concatenate([sample, rng.normal(10.0, 1.0, (15, 4))])

    out_a = bn(batch_a)[0]
    out_b = bn(batch_b)[0]

    assert not np.allclose(out_a, out_b, atol=1e-3), (
        "训练模式下同一个样本在不同 batch 中输出应当不同"
    )


def test_eval_output_is_deterministic_across_batches() -> None:
    """推理时同一个样本的输出必须**与同 batch 的其它样本无关**。

    这是线上服务的基本要求：不能因为并发批次的组成不同而给出不同推荐。
    """
    rng = np.random.default_rng(3)
    sample = rng.normal(5.0, 3.0, (1, 4))

    bn = ManualBatchNorm1d(4)
    bn.training = True
    bn(make_input(rows=32, seed=4))
    bn.training = False

    batch_a = np.concatenate([sample, rng.normal(0.0, 1.0, (15, 4))])
    batch_b = np.concatenate([sample, rng.normal(10.0, 1.0, (15, 4))])

    np.testing.assert_allclose(bn(batch_a)[0], bn(batch_b)[0], atol=1e-12)
