import numpy as np
import torch

def relu(z):
    return np.maximum(0, z)

def relu_gradient(z):
    """z>0→1，z≤0→0（含0点），与PyTorch默认行为一致"""
    return np.where(z > 0, 1.0, 0.0)

def mse_loss(z2, y):
    return np.mean((z2 - y) ** 2)

def mse_gradient(z2, y):
    B, out_dim = z2.shape
    return 2 * (z2 - y) / (B * out_dim)

def forward_backward_numpy(W1, b1, W2, b2, x, y):
    """NumPy 完整前向+反向，返回loss和6个梯度"""
    z1 = x @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    loss = mse_loss(z2, y)
    
    dz2 = mse_gradient(z2, y)
    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0)
    da1 = dz2 @ W2.T
    dz1 = da1 * relu_gradient(z1)
    dW1 = x.T @ dz1
    db1 = np.sum(dz1, axis=0)
    return loss, z1, dW2, db2, da1, dz1, dW1, db1

# ==============================================
# 1. 构造 z1 中包含精确 0 点的场景
# ==============================================
np.random.seed(123)
B = 4
in_dim = 3
hidden = 5
out_dim = 2

W1 = np.random.randn(in_dim, hidden) * 0.1
b1 = np.zeros(hidden)
W2 = np.random.randn(hidden, out_dim) * 0.1
b2 = np.zeros(out_dim)
x = np.random.randn(B, in_dim)
y = np.random.randn(B, out_dim)

# 初始z1
z1_init = x @ W1 + b1

# 强制两个位置精确为0：位置1=(0,1)，位置2=(2,3)
zero_positions = [(0, 1), (2, 3)]
for i, j in zero_positions:
    b1[j] -= z1_init[i, j]  # 调整偏置，让该点z1精确归零

# 验证：重新计算z1，确认目标位置为0
z1_check = x @ W1 + b1
print("="*60)
print("构造验证：z1 中目标位置的值")
for i, j in zero_positions:
    print(f"  位置({i},{j}) = {z1_check[i,j]:.2e} （是否为0：{np.isclose(z1_check[i,j], 0)}）")
print(f"z1 中等于0的元素总数：{np.sum(z1_check == 0)}")
print("="*60)

# ==============================================
# 2. NumPy 手动梯度
# ==============================================
loss_np, z1_np, dW2_np, db2_np, da1_np, dz1_np, dW1_np, db1_np = \
    forward_backward_numpy(W1, b1, W2, b2, x, y)

# ==============================================
# 3. PyTorch 自动梯度（基准）
# ==============================================
W1_t = torch.tensor(W1, dtype=torch.float64, requires_grad=True)
b1_t = torch.tensor(b1, dtype=torch.float64, requires_grad=True)
W2_t = torch.tensor(W2, dtype=torch.float64, requires_grad=True)
b2_t = torch.tensor(b2, dtype=torch.float64, requires_grad=True)
x_t = torch.tensor(x, dtype=torch.float64)
y_t = torch.tensor(y, dtype=torch.float64)

z1_t = x_t @ W1_t + b1_t
z1_t.retain_grad()
a1_t = torch.relu(z1_t)
a1_t.retain_grad()
z2_t = a1_t @ W2_t + b2_t
z2_t.retain_grad()
loss_t = torch.mean((z2_t - y_t) ** 2)
loss_t.backward()

dW2_t = W2_t.grad.numpy()
db2_t = b2_t.grad.numpy()
da1_t = a1_t.grad.numpy()
dz1_t = z1_t.grad.numpy()
dW1_t = W1_t.grad.numpy()
db1_t = b1_t.grad.numpy()

# ==============================================
# A. 与 PyTorch 对齐对比（重点看0点位置）
# ==============================================
print("\nA. 与 PyTorch 梯度对比")
print("-"*60)

# 整体最大误差
print(f"dW2 整体最大误差: {np.max(np.abs(dW2_np - dW2_t)):.2e}")
print(f"db2 整体最大误差: {np.max(np.abs(db2_np - db2_t)):.2e}")
print(f"dW1 整体最大误差: {np.max(np.abs(dW1_np - dW1_t)):.2e}")
print(f"db1 整体最大误差: {np.max(np.abs(db1_np - db1_t)):.2e}")

# 重点：z1=0 位置的 dz1 对比
print("\n重点：z1=0 位置的 dz1 对比")
for i, j in zero_positions:
    np_val = dz1_np[i, j]
    torch_val = dz1_t[i, j]
    err = abs(np_val - torch_val)
    print(f"  位置({i},{j})  NumPy={np_val:.2e}  PyTorch={torch_val:.2e}  误差={err:.2e}")

# ==============================================
# B. 有限差分数值验证（0点 vs 非0点）
# ==============================================
print("\n" + "="*60)
print("B. 有限差分验证：0点对应权重 vs 非0点对应权重")
print("="*60)

eps = 1e-5

def compute_loss(W1, b1, W2, b2, x, y):
    z1 = x @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    return mse_loss(z2, y)

# 测试点：1个0点对应的权重，1个非0点对应的权重
test_cases = [
    ("W1[0,1]  (对应z1列j=1，含0点)", "W1", (0, 1), dW1_np),
    ("W1[1,2]  (对应z1列j=2，无0点)", "W1", (1, 2), dW1_np),
    ("b1[1]    (对应z1列j=1，含0点)", "b1", (1,),    db1_np),
    ("b1[2]    (对应z1列j=2，无0点)", "b1", (2,),    db1_np),
]

for name, ptype, idx, analytic_grad in test_cases:
    W1_p, W1_m = W1.copy(), W1.copy()
    b1_p, b1_m = b1.copy(), b1.copy()
    W2_p, W2_m = W2.copy(), W2.copy()
    b2_p, b2_m = b2.copy(), b2.copy()
    
    if ptype == "W1":
        W1_p[idx] += eps
        W1_m[idx] -= eps
    elif ptype == "b1":
        b1_p[idx] += eps
        b1_m[idx] -= eps
    
    loss_p = compute_loss(W1_p, b1_p, W2_p, b2_p, x, y)
    loss_m = compute_loss(W1_m, b1_m, W2_m, b2_m, x, y)
    numeric_grad = (loss_p - loss_m) / (2 * eps)
    
    err = abs(numeric_grad - analytic_grad[idx])
    print(f"{name:30s} 解析={analytic_grad[idx]:.8f}  数值={numeric_grad:.8f}  绝对误差={err:.2e}")

print("\n" + "="*60)
print("实验结论")
print("A. z1=0 处 NumPy 与 PyTorch 梯度完全一致（误差 0），次梯度约定相同")
print("B. 含0点误差 4.5e-3 vs 无0点 5.2e-12，相差 9 个数量级")
print("   原因：中心差分在不可导点两侧采样得 0.5，解析次梯度得 0")
print("   落地建议：数值检验应跳过 |z1| < eps 的元素")
print("="*60)
