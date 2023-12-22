# 验证实验结果
import numpy as np
import torch
torch.random.manual_seed(0)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
a = np.array([2, 4, 8], dtype=np.float64).reshape(3, 1)
w = np.array([0.1,0.2,0.3,0.4,0.5,0.6], dtype=np.float64).reshape(2, 3)
b = np.array([0.1, 0.5], dtype=np.float64).reshape(2, 1)
r = w @ a + b
# print(a)
print(w)
# print(b)
print(r)
print(sigmoid(r))