import torch

# ========== 实验1：标量求导 y = x² ==========
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
y.backward()
print("实验1  x.grad =", x.grad.item())
# 观察：输出 6.0，和手算导数 dy/dx = 2x = 2×3 = 6 完全吻合，验证标量求导正确

# ========== 实验2：复合函数链式法则 y = (2x + 1)² ==========
x = torch.tensor(1.0, requires_grad=True)
y = (2 * x + 1) ** 2
y.backward()
print("实验2  x.grad =", x.grad.item())
# 观察：输出 12.0，链式法则手算：dy/dx = 2·(2x+1)·2 = 4×(2×1+1) = 12，验证链式法则自动生效

# ========== 实验3：多变量偏导 z = x·y + x² ==========
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = x * y + x ** 2
z.backward()
print(f"实验3  x.grad = {x.grad.item()},  y.grad = {y.grad.item()}")
# 观察：x.grad=7.0（∂z/∂x = y + 2x = 3+4 = 7），y.grad=2.0（∂z/∂y = x = 2），两个梯度分别对应各自偏导数

# ========== 实验4：梯度累加效应 ==========
x = torch.tensor(3.0, requires_grad=True)
# 第一次反向传播
y1 = x ** 2
y1.backward()
print("实验4  第1次反向后 x.grad =", x.grad.item())
# 第二次反向传播，不调用 zero_grad()
y2 = x ** 2
y2.backward()
print("实验4  第2次反向后 x.grad =", x.grad.item())
# 观察：第一次 6.0，第二次变成 12.0，说明梯度是「累加」而非「覆盖」，这就是训练时必须调用 optimizer.zero_grad() 的原因

# ========== 实验5：sigmoid 梯度消失 ==========
x = torch.tensor(10.0, requires_grad=True)
y = torch.sigmoid(x)
y.backward()
print("实验5  sigmoid(10) 的梯度 =", x.grad.item())
# 观察：梯度约等于 0（极小值），因为 sigmoid'(x) = sigmoid(x)·(1-sigmoid(x))，x 很大时 sigmoid≈1，导数≈0，这就是深层网络梯度消失的来源之一
