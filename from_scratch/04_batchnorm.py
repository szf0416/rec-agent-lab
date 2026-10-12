"""BatchNorm：手写实现 + 训练/推理行为验证。

对应 S1 验收标准第 3 条的一半：

    能回答：BatchNorm 训练/推理为什么不同？

## 为什么值得单独做

ROADMAP 的 S1-W4 把 batchnorm 列为正则化对照之一，但我们前几周只做了
L2 / dropout，跳过了它。而 BN 有个**极易答错**的点：

    「为什么训练和推理的行为不一样？」

多数人答"因为要归一化"。真正的原因有三层，见文件末尾的结论。

## 三个必须理解的机制

1. **训练**：用**当前 batch** 的均值/方差归一化
2. **推理**：用**训练期累积的滑动平均**统计量归一化
3. **为什么必须分开**：推理时可能只有 1 个样本（batch=1 时方差为 0），
   没法算 batch 统计；而且线上请求是逐条来的，行为必须确定

用法::

    python from_scratch/04_batchnorm.py
"""

from __future__ import annotations

import numpy as np
import torch

# --------------------------------------------------------------------------
# 手写 BatchNorm1d
# --------------------------------------------------------------------------


class ManualBatchNorm1d:
    """纯 NumPy 的 BatchNorm1d。

    公式：

        train:  y = gamma * (x - mean_batch) / sqrt(var_batch + eps) + beta
        eval:   y = gamma * (x - running_mean) / sqrt(running_var + eps) + beta

    两个容易写错的细节（下面的实现都处理了）：

    **细节 1：方差的有偏/无偏之分。**
    PyTorch 的 BN 在**归一化**时用**有偏**方差（除以 N），
    但**滑动平均里存的是无偏**方差（除以 N-1）。两者不一致是刻意的：
    归一化关心当前 batch 的真实二阶矩，而滑动平均是对总体方差的无偏估计。

    **细节 2：滑动平均的更新用的是 batch 统计，不是归一化后的值。**
    顺序必须是「先用当前 batch 算统计 → 更新 running → 再归一化」
    （顺序对结果无影响，但读代码时容易搞混）。

    Args:
        num_features: 特征维度（BN 在每个通道/特征上独立归一化）。
        eps: 防止除零的小量。PyTorch 默认 1e-5。
        momentum: 滑动平均的更新率。**注意 PyTorch 的语义**：
            `running = (1 - momentum) * running + momentum * batch`
            默认 0.1（即新值占 10%）。
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ) -> None:
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # 可学习参数：gamma 缩放、beta 平移（初始为恒等变换）
        self.gamma = np.ones(num_features, dtype=np.float64)
        self.beta = np.zeros(num_features, dtype=np.float64)

        # 滑动平均统计量：训练期累积，推理期使用
        self.running_mean = np.zeros(num_features, dtype=np.float64)
        self.running_var = np.ones(num_features, dtype=np.float64)

        self.training = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.training:
            mean = x.mean(axis=0)
            # 有偏方差（ddof=0）：用于归一化
            var_biased = x.var(axis=0)
            # 无偏方差（ddof=1）：用于更新滑动平均
            batch_size = x.shape[0]
            var_unbiased = var_biased * batch_size / max(batch_size - 1, 1)

            self.running_mean = (
                1.0 - self.momentum
            ) * self.running_mean + self.momentum * mean
            self.running_var = (
                1.0 - self.momentum
            ) * self.running_var + self.momentum * var_unbiased

            normalized = (x - mean) / np.sqrt(var_biased + self.eps)
        else:
            normalized = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

        return self.gamma * normalized + self.beta

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            "gamma": self.gamma,
            "beta": self.beta,
            "running_mean": self.running_mean.copy(),
            "running_var": self.running_var.copy(),
        }


# --------------------------------------------------------------------------
# 验证
# --------------------------------------------------------------------------


def compare_train_mode() -> None:
    """训练模式：手写实现必须与 PyTorch 一致。"""
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 3.0, (16, 4))  # 故意用非 0 均值、非 1 方差

    manual = ManualBatchNorm1d(4)
    manual.training = True
    out_manual = manual(x)

    # 必须 .double()：BN 的 weight/bias 默认是 float32，
    # 与 float64 输入混用会报 "mixed dtype (CPU): all inputs must share same datatype"。
    torch_bn = torch.nn.BatchNorm1d(4, eps=1e-5, momentum=0.1).double()
    torch_bn.train()
    out_torch = torch_bn(torch.tensor(x, dtype=torch.float64)).detach().numpy()

    error = float(np.max(np.abs(out_manual - out_torch)))
    print("【训练模式】手写 vs PyTorch")
    print(f"  最大绝对误差      : {error:.3e}")
    print(f"  输出均值（应≈0）  : {out_manual.mean(axis=0).round(6)}")
    print(f"  输出标准差（应≈1）: {out_manual.std(axis=0).round(6)}")
    print(f"  一致              : {error < 1e-10}")
    print()

    # 顺手验证「滑动平均存无偏方差」这个细节
    rm_error = float(np.max(np.abs(manual.running_mean - torch_bn.running_mean.numpy())))
    rv_error = float(np.max(np.abs(manual.running_var - torch_bn.running_var.numpy())))
    print("  滑动平均统计量对比")
    print(f"    running_mean 误差: {rm_error:.3e}")
    print(f"    running_var  误差: {rv_error:.3e}   ← 这里验证了「无偏方差」细节")
    print(f"    一致: {rm_error < 1e-10 and rv_error < 1e-10}")
    print()


def compare_eval_mode() -> None:
    """推理模式：用滑动平均统计量，也要与 PyTorch 一致。"""
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 3.0, (16, 4))

    manual = ManualBatchNorm1d(4)
    manual.training = True
    manual(x)  # 先跑一次训练，累积统计量
    manual.training = False
    out_manual = manual(x)

    # 必须 .double()：BN 的 weight/bias 默认是 float32，
    # 与 float64 输入混用会报 "mixed dtype (CPU): all inputs must share same datatype"。
    torch_bn = torch.nn.BatchNorm1d(4, eps=1e-5, momentum=0.1).double()
    torch_bn.train()
    torch_bn(torch.tensor(x, dtype=torch.float64))
    torch_bn.eval()
    out_torch = torch_bn(torch.tensor(x, dtype=torch.float64)).detach().numpy()

    error = float(np.max(np.abs(out_manual - out_torch)))
    print("【推理模式】手写 vs PyTorch")
    print(f"  最大绝对误差: {error:.3e}")
    print(f"  一致        : {error < 1e-10}")
    print()


def demonstrate_train_eval_difference() -> None:
    """核心演示：同一个输入，训练与推理的输出**完全不同**。

    这是回答「BN 训练/推理为什么不同」最直观的证据。
    """
    rng = np.random.default_rng(1)
    x = rng.normal(5.0, 3.0, (16, 4))

    bn = ManualBatchNorm1d(4)

    bn.training = True
    out_train = bn(x)
    bn.training = False
    out_eval = bn(x)

    print("【关键演示】同一输入，训练 vs 推理")
    print(f"  输入均值: {x.mean(axis=0).round(4)}")
    print(f"  训练输出均值: {out_train.mean(axis=0).round(4)}   ← 被归一化到 ≈0")
    print(f"  推理输出均值: {out_eval.mean(axis=0).round(4)}   ← 用的是 running_mean，不是 0")
    print()
    print(f"  两种模式输出的最大差异: {np.max(np.abs(out_train - out_eval)):.4f}")
    print("  → 同一个输入、同一组参数，输出不同。这就是「训练/推理行为不同」。")
    print()


def demonstrate_why_eval_uses_running_stats() -> None:
    """为什么推理不能用 batch 统计：batch=1 时方差为 0，无法归一化。

    这是"必须用滑动平均"最硬的理由。
    """
    x_single = np.array([[5.0, 3.0, -1.0, 2.0]])  # 只有 1 个样本

    print("【为什么推理必须用滑动平均】场景：只有 1 个样本")
    print(f"  输入: {x_single[0]}")

    # 若强行用 batch 统计
    mean = x_single.mean(axis=0)
    var = x_single.var(axis=0)
    print(f"  batch 均值: {mean}   （等于输入本身）")
    print(f"  batch 方差: {var}   ← 全 0！")
    with np.errstate(divide="ignore", invalid="ignore"):
        broken = (x_single - mean) / np.sqrt(var + 1e-5)
    print(f"  强行归一化结果: {np.round(broken, 6)[0]}   ← 全是 0，信息丢失")
    print()

    # 用滑动平均就没问题
    bn = ManualBatchNorm1d(4)
    bn.training = True
    bn(np.random.default_rng(0).normal(5.0, 3.0, (32, 4)))  # 用历史 batch 累积统计
    bn.training = False
    out = bn(x_single)
    print(f"  用滑动平均归一化: {np.round(out, 6)[0]}")
    print(f"  → 仍然是有意义的输出（因为 running_var 来自历史 batch，不为 0）")
    print()


def demonstrate_batch_statistics_vs_global() -> None:
    """训练用 batch 统计 → 输出依赖 batch 组成；推理固定 → 输出确定。

    这解释了另一个常见困惑：为什么训练时同一个样本在不同 batch 里输出不同。
    """
    rng = np.random.default_rng(2)
    sample = rng.normal(5.0, 3.0, (1, 4))

    bn = ManualBatchNorm1d(4)
    bn.training = True

    # 把同一个样本放进两个不同的 batch
    batch_a = np.concatenate([sample, rng.normal(0.0, 1.0, (15, 4))])
    batch_b = np.concatenate([sample, rng.normal(10.0, 1.0, (15, 4))])

    out_a = bn(batch_a)[0]
    out_b = bn(batch_b)[0]

    print("【训练时输出依赖 batch 组成】同一个样本，放入两个不同 batch")
    print(f"  batch A（其他样本均值≈0）:  该样本输出 = {out_a.round(4)}")
    print(f"  batch B（其他样本均值≈10）: 该样本输出 = {out_b.round(4)}")
    print(f"  两者差异: {np.max(np.abs(out_a - out_b)):.4f}")
    print("  → 训练时 BN 的输出依赖同 batch 的其它样本，这是它的特性也是它的风险")
    print()


def main() -> int:
    print("=" * 72)
    print("BatchNorm：手写实现与行为验证")
    print("=" * 72)
    print()

    compare_train_mode()
    compare_eval_mode()
    demonstrate_train_eval_difference()
    demonstrate_why_eval_uses_running_stats()
    demonstrate_batch_statistics_vs_global()

    print("=" * 72)
    print("结论")
    print("=" * 72)
    print()
    print("Q: BatchNorm 训练/推理为什么不同？")
    print()
    print("三层回答（面试按这个顺序说）：")
    print()
    print("  1. 【机制】训练用当前 batch 的均值/方差归一化；")
    print("     推理用训练期累积的滑动平均（running_mean / running_var）。")
    print()
    print("  2. 【为什么必须分开】推理时可能只有 1 个样本 ——")
    print("     batch 方差恒为 0，强行归一化会把输出全变成 0、信息丢失。")
    print("     而且线上请求是逐条来的，行为必须确定、不能依赖同批次的其它样本。")
    print()
    print("  3. 【实现细节】滑动平均存的是**无偏**方差（除以 N-1），")
    print("     而归一化用的是**有偏**方差（除以 N）。")
    print("     PyTorch 实测证明这一点（本脚本已对齐验证）。")
    print()
    print("顺带理解的两个现象：")
    print("  - 训练时同一个样本在不同 batch 里输出不同（因为 batch 统计变了）")
    print("  - 忘记 model.eval() 会让验证结果不稳定，因为还在用 batch 统计")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
