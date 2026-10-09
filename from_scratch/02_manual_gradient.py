import numpy as np
# 定义ReLU激活函数
def relu(x):
    return np.maximum(0, x)

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def mse_gradient(z2, y):
    
    B,out_dim = z2.shape
    return 2 * (z2 - y) / (B * out_dim)

def relu_gradient(z):
    return np.where(z > 0, 1.0, 0.0)

np.random.seed(42)

B = 4
in_dim = 3
hidden = 5
out_dim = 2

W1 = np.random.randn(in_dim, hidden)*0.1
b1 = np.zeros(hidden)
W2 = np.random.randn(hidden, out_dim)*0.1
b2 = np.zeros(out_dim)

x = np.random.randn(B, in_dim)
y = np.random.randn(B, out_dim)
# Forward
z1 = x @ W1 + b1
a1 = relu(z1)
z2 = a1 @ W2 + b2
loss = mse_loss(z2, y)
print("loss=",loss)
print("="*50)

# Backward
dz2 = mse_gradient(z2, y)  #B*out_dim
dw2 = a1.T @ dz2    #hidden*out_dim
da1 = dz2 @ W2.T  #B*hidden
db1 = np.sum(da1, axis=0)  #hidden
dz1 = da1 * relu_gradient(z1) #B*hidden
dw1 = x.T @ dz1    #in_dim*hidden
db2 = np.sum(dz2, axis=0) #out_dim

# ==============================================
# 5. 打印结果验证形状
# ==============================================
print("6个梯度形状验证：")
print(f"dW2 形状: {dw2.shape}  (应与W2一致: {W2.shape})")
print(f"db2 形状: {db2.shape}  (应与b2一致: {b2.shape})")
print(f"da1 形状: {da1.shape}  (应与a1一致: {a1.shape})")
print(f"dz1 形状: {dz1.shape}  (应与z1一致: {z1.shape})")
print(f"dW1 形状: {dw1.shape}  (应与W1一致: {W1.shape})")
print(f"db1 形状: {db1.shape}  (应与b1一致: {b1.shape})")

