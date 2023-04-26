import torch
import numpy as np

x=torch.randn(5)
print(x.abs())
print(x.abs().sqrt())
print(torch.Tensor(3,4))