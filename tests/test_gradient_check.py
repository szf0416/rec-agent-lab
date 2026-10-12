"""手写两层 MLP 的梯度验证（进 CI）。

对应 S1 验收标准第 2 条：

    手写版本与 PyTorch 版本梯度误差 < 1e-6（**有测试断言，进 CI**）

## 为什么教学脚本之外还要单独写测试

`from_scratch/03_gradient_check.py` 是演算纸 —— 靠人眼看输出对不对，
而且它在 ruff / mypy / pytest 的排除列表里，**根本不进 CI**。
所以验收标准里"有测试断言，进 CI"这一条，光有脚本是不成立的。

本文件把同样的验证变成断言，于是：
1. 将来谁改坏了梯度公式，CI 立刻红
2. 验收标准那条才真正落地

## 为什么在测试里重新实现，而不是 import from_scratch

- `from_scratch/` 不是 Python 包，文件名以数字开头，无法正常 import
- 用 `importlib` 动态加载会执行脚本顶层代码（打印一大堆输出），污染测试结果

所以这里独立实现一份。好处是测试**不依赖教学脚本**，更稳定；
代价是公式有两份 —— 但这本身也是一种交叉验证。

## 与 W3/W4 的关系

本文件只验证「手写梯度是否正确」。
过拟合/正则化/优化器的验证在各自实验脚本与报告里，不重复。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

# --------------------------------------------------------------------------
# 纯 NumPy 手写实现（不使用任何自动微分）
# --------------------------------------------------------------------------

FEATURE_DIM = 4
HIDDEN_DIM = 5
OUTPUT_DIM = 2


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU 前向。

    用 np.maximum 而不是原地赋值：原地写法会破坏调用方的数组，
    详见 S1-W1 评审记录里修过的那个 bug。
    """
    return np.maximum(0.0, x)


def relu_gradient(z: np.ndarray) -> np.ndarray:
    """ReLU 次梯度：z > 0 → 1，z ≤ 0 → 0（含 0 点，与 PyTorch 默认一致）。

    注意 0 点处不可导，这是工程约定；PyTorch 实测 `z=0.0 -> grad=0.0`。
    这个约定会让**数值梯度检验在 0 点附近失效**（见 test_numberical_gradient_*）。
    """
    return np.where(z > 0, 1.0, 0.0)


def forward(
    x: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """前向：z1 = xW1 + b1 → a1 = relu(z1) → z2 = a1W2 + b2。"""
    z1 = x @ w1 + b1
    a1 = relu(z1)
    z2 = a1 @ w2 + b2
    return z1, a1, z2


def mse_loss(prediction: np.ndarray, target: np.ndarray) -> float:
    """均方误差。

    注意签名只依赖入参，不引用任何外部变量 ——
    S1-W1 评审曾修过一个"函数内部引用全局 z2"的 bug。
    """
    return float(np.mean((prediction - target) ** 2))


def manual_backward(
    x: np.ndarray,
    y: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    z1: np.ndarray,
    a1: np.ndarray,
    z2: np.ndarray,
) -> dict[str, np.ndarray]:
    """手写反向传播，返回全部 6 个梯度。

    推导（维度必须自洽，写错时形状会立刻暴露）：

        loss = mean((z2 - y)²)
        dz2  = 2(z2 - y) / (B · out_dim)     (B, out_dim)
        dW2  = a1ᵀ @ dz2                     (hidden, out_dim)
        db2  = sum(dz2, axis=0)              (out_dim,)
        da1  = dz2 @ W2ᵀ                     (B, hidden)
        dz1  = da1 ⊙ relu'(z1)                (B, hidden)
        dW1  = xᵀ @ dz1                      (in_dim, hidden)
        db1  = sum(dz1, axis=0)              (hidden,)
    """
    batch, out_dim = z2.shape

    dz2 = 2.0 * (z2 - y) / (batch * out_dim)
    dw2 = a1.T @ dz2
    db2 = dz2.sum(axis=0)
    da1 = dz2 @ w2.T
    dz1 = da1 * relu_gradient(z1)
    dw1 = x.T @ dz1
    db1 = dz1.sum(axis=0)

    return {"dw1": dw1, "db1": db1, "dw2": dw2, "db2": db2, "da1": da1, "dz1": dz1}


# --------------------------------------------------------------------------
# 测试夹具：确定性参数与数据
# --------------------------------------------------------------------------


def make_problem(
    batch: int = 8, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """构造确定性的小规模问题。

    全部用 float64：float32 的机器精度约 1e-7，
    与验收阈值 1e-6 太接近，容易出现"偶尔失败"的假阳性。
    """
    rng = np.random.default_rng(seed)
    params = {
        "w1": rng.normal(0.0, 0.5, (FEATURE_DIM, HIDDEN_DIM)),
        "b1": rng.normal(0.0, 0.1, HIDDEN_DIM),
        "w2": rng.normal(0.0, 0.5, (HIDDEN_DIM, OUTPUT_DIM)),
        "b2": rng.normal(0.0, 0.1, OUTPUT_DIM),
    }
    x = rng.normal(0.0, 1.0, (batch, FEATURE_DIM))
    y = rng.normal(0.0, 1.0, (batch, OUTPUT_DIM))
    return x, y, params


def to_torch(array: np.ndarray, *, requires_grad: bool = False) -> torch.Tensor:
    return torch.tensor(array, dtype=torch.float64, requires_grad=requires_grad)


# --------------------------------------------------------------------------
# 1. 前向一致性
# --------------------------------------------------------------------------


def test_forward_matches_pytorch() -> None:
    """手写前向必须与 PyTorch 逐元素一致。

    先验证前向是有意为之：前向不对的话，梯度对不上时你分不清是哪一步的问题。
    """
    # 只验证前向，不需要标签
    x, _, params = make_problem()
    _, _, z2_manual = forward(x, params["w1"], params["b1"], params["w2"], params["b2"])

    x_t = to_torch(x)
    z2_torch = torch.relu(x_t @ to_torch(params["w1"]) + to_torch(params["b1"])) @ to_torch(
        params["w2"]
    ) + to_torch(params["b2"])
    torch.testing.assert_close(to_torch(z2_manual), z2_torch)


# --------------------------------------------------------------------------
# 2. 六个梯度与 PyTorch 对齐（验收标准的核心断言）
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["dw1", "db1", "dw2", "db2", "da1", "dz1"])
def test_manual_gradient_matches_pytorch(name: str) -> None:
    """手写梯度与 PyTorch 自动微分的最大绝对误差必须 < 1e-6。

    实际用 float64 时误差在 1e-17 量级（浮点舍入的极限），
    阈值 1e-6 留了充足余量。
    """
    x, y, params = make_problem()
    z1, a1, z2 = forward(x, params["w1"], params["b1"], params["w2"], params["b2"])
    manual = manual_backward(
        x, y, params["w1"], params["b1"], params["w2"], params["b2"], z1, a1, z2
    )

    x_t = to_torch(x)
    y_t = to_torch(y)
    w1_t = to_torch(params["w1"], requires_grad=True)
    b1_t = to_torch(params["b1"], requires_grad=True)
    w2_t = to_torch(params["w2"], requires_grad=True)
    b2_t = to_torch(params["b2"], requires_grad=True)

    z1_t = x_t @ w1_t + b1_t
    z1_t.retain_grad()
    a1_t = torch.relu(z1_t)
    a1_t.retain_grad()
    z2_t = a1_t @ w2_t + b2_t
    torch.mean((z2_t - y_t) ** 2).backward()

    torch_map = {
        "dw1": w1_t.grad,
        "db1": b1_t.grad,
        "dw2": w2_t.grad,
        "db2": b2_t.grad,
        "da1": a1_t.grad,
        "dz1": z1_t.grad,
    }
    assert torch_map[name] is not None, f"{name} 的 PyTorch 梯度为 None"

    error = float(np.max(np.abs(manual[name] - torch_map[name].numpy())))
    assert error < 1e-6, f"{name} 与 PyTorch 的误差 {error:.3e} 超过阈值 1e-6"


def test_all_gradients_have_correct_shapes() -> None:
    """每个梯度必须与对应参数形状一致 —— 形状错了，公式一定错了。"""
    x, y, params = make_problem()
    z1, a1, z2 = forward(x, params["w1"], params["b1"], params["w2"], params["b2"])
    manual = manual_backward(
        x, y, params["w1"], params["b1"], params["w2"], params["b2"], z1, a1, z2
    )

    assert manual["dw1"].shape == params["w1"].shape
    assert manual["db1"].shape == params["b1"].shape
    assert manual["dw2"].shape == params["w2"].shape
    assert manual["db2"].shape == params["b2"].shape
    assert manual["da1"].shape == a1.shape
    assert manual["dz1"].shape == z1.shape


# --------------------------------------------------------------------------
# 3. 数值梯度检验（独立于 PyTorch 的第二重验证）
# --------------------------------------------------------------------------


def _numeric_gradient(
    params: dict[str, np.ndarray],
    key: str,
    index: tuple[int, ...],
    x: np.ndarray,
    y: np.ndarray,
    *,
    eps: float = 1e-5,
) -> float:
    """中心差分：∂L/∂θ ≈ (L(θ+ε) - L(θ-ε)) / 2ε。

    中心差分的截断误差是 O(ε²)，前向差分是 O(ε)。
    ε 取 1e-5 是截断误差与浮点误差的平衡点：
    - ε 太大（1e-2）→ 截断误差大
    - ε 太小（1e-12）→ 浮点相减出现灾难性抵消，结果变成噪声
    """
    plus = {k: v.copy() for k, v in params.items()}
    minus = {k: v.copy() for k, v in params.items()}
    plus[key][index] += eps
    minus[key][index] -= eps

    def loss_of(p: dict[str, np.ndarray]) -> float:
        _, _, z2 = forward(x, p["w1"], p["b1"], p["w2"], p["b2"])
        return mse_loss(z2, y)

    return (loss_of(plus) - loss_of(minus)) / (2 * eps)


@pytest.mark.parametrize(
    ("key", "index"),
    [
        ("w1", (0, 0)),
        ("w1", (2, 3)),
        ("b1", (1,)),
        ("w2", (0, 1)),
        ("w2", (3, 0)),
        ("b2", (0,)),
        ("b2", (1,)),
    ],
)
def test_numeric_gradient_agrees_with_manual(key: str, index: tuple[int, ...]) -> None:
    """数值梯度（中心差分）与手写解析梯度必须一致。

    这是**独立于 PyTorch 的第二重验证**：如果 PyTorch 自身实现有偏差，
    上面那组「与 PyTorch 对齐」的测试会一起错，而这里不会。
    """
    x, y, params = make_problem()
    z1, a1, z2 = forward(x, params["w1"], params["b1"], params["w2"], params["b2"])
    manual = manual_backward(
        x, y, params["w1"], params["b1"], params["w2"], params["b2"], z1, a1, z2
    )

    numeric = _numeric_gradient(params, key, index, x, y)
    analytic = float(manual[f"d{key}"][index])
    assert abs(numeric - analytic) < 1e-6, f"{key}{index}: 解析={analytic:.8f} 数值={numeric:.8f}"


def test_numeric_gradient_fails_at_non_differentiable_point() -> None:
    """在 ReLU 的不可导点（z = 0）处，中心差分与解析次梯度相差 0.5。

    这不是 bug，而是中心差分在不可导点的固有困难：

        (relu(ε) - relu(-ε)) / (2ε) = (ε - 0) / (2ε) = 0.5
        而解析次梯度（本项目约定 z=0 处为 0）  = 0

    实测（S1-W1 的边界实验）：
        含 z1=0 的列：误差 4.50e-03
        无 z1=0 的列：误差 5.21e-12
        **相差 9 个数量级**

    所以做数值梯度检验时应跳过 `|z| < ε` 的元素，或接受这个已知差异。

    ## 实现上踩过的坑（值得记住）

    第一版我把这个验证写成了"在网络里扰动 b1，看数值梯度是否显著偏离 0"，
    结果只得到 **5e-06** 而不是 0.5 —— 因为损失函数是 **MSE（平方）**：

        loss = relu(z1)²  →  数值梯度 = ε²/(2ε) = ε/2 = 5e-6

    **平方把 ε 压缩掉了**，不可导点的影响被淹没。

    所以这里改成**直接对 ReLU 做中心差分**，隔离掉损失函数的影响，
    才能看到 0.5 这个本质差异。
    """
    eps = 1e-5

    # --- 直接对 ReLU 求中心差分，展示数学本质 ---
    numeric = float((relu(np.array([eps]))[0] - relu(np.array([-eps]))[0]) / (2 * eps))
    analytic = float(relu_gradient(np.array([0.0]))[0])

    assert analytic == 0.0, "次梯度约定：z=0 处应为 0（与 PyTorch 实测一致）"
    assert numeric == pytest.approx(0.5, abs=1e-6), "中心差分在 0 点给出 0.5"
    assert abs(numeric - analytic) == pytest.approx(0.5, abs=1e-6), (
        "不可导点处数值梯度与解析梯度的差异正好是 0.5"
    )

    # --- 网络层面：z1 精确为 0 时，该处解析梯度被 ReLU 的 0 次梯度门控掉 ---
    w1 = np.zeros((1, 1))
    b1 = np.zeros(1)
    w2 = np.ones((1, 1))
    b2 = np.zeros(1)
    x = np.zeros((1, 1))  # z1 = 0·w1 + 0 = 0，正落在不可导点
    y = np.zeros((1, 1))

    z1, a1, z2 = forward(x, w1, b1, w2, b2)
    assert z1[0, 0] == 0.0, "构造失败：z1 应为精确的 0"

    manual = manual_backward(x, y, w1, b1, w2, b2, z1, a1, z2)
    assert manual["dz1"][0, 0] == pytest.approx(0.0), "不可导点处解析梯度为 0"
    assert manual["db1"][0] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# 4. 与损失函数实现的一致性
# --------------------------------------------------------------------------


def test_loss_matches_pytorch() -> None:
    """手写 MSE 必须与 PyTorch 的 mean((pred-target)²) 一致。"""
    x, y, params = make_problem()
    _, _, z2 = forward(x, params["w1"], params["b1"], params["w2"], params["b2"])

    manual = mse_loss(z2, y)
    torch_value = float(torch.mean((to_torch(z2) - to_torch(y)) ** 2))
    assert manual == pytest.approx(torch_value, abs=1e-12)


def test_loss_does_not_depend_on_external_state() -> None:
    """损失函数必须只依赖入参 —— 防止"函数内部引用全局变量"那类 bug 回归。

    测试方法：先算一次，然后把入参换成完全不同的值再算，
    结果必须跟着变。如果函数引用了某个模块级变量，第二次会算出错误结果。
    """
    rng = np.random.default_rng(0)
    a = rng.normal(size=(4, 2))
    b = rng.normal(size=(4, 2))
    first = mse_loss(a, b)

    c = rng.normal(size=(4, 2)) + 10.0
    second = mse_loss(c, b)

    assert first != second
    assert second == pytest.approx(float(np.mean((c - b) ** 2)))
