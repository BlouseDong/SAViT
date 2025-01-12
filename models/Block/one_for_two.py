import torch
import torch.nn as nn
from einops import rearrange


class Onefortwo(nn.Module):
    def __init__(self, H=24, W=24, hh=4, ww=4, box_num=3, batch=16, n=16):
        super(Onefortwo, self).__init__()
        self.H = H
        self.W = W
        self.hh = hh
        self.ww = ww
        self.box_num = box_num
        self.batch = batch
        self.n = n

    def forward(self, x):
        B, N, C = x.shape  # B: 批大小, N: 特征数, C: 通道数

        # Divide x into x_restored and y
        x_restored = x[:, :576, :]
        y = x[:, 576:, :]

        # Reshape x_restored to (B, C, H, W)
        x_restored = x_restored.permute(0, 2, 1).reshape(B, C, self.H, self.W)

        # Rearrange y to (box_num, batch, n, d)
        y_restored = rearrange(y, 'batch (box_num n) d -> box_num batch n d', box_num=self.box_num, n=self.n)

        return x_restored, y_restored


# # Example usage:
# # Initialize the module
# module = Onefortwo()
# x = torch.load(r"D:\zhiwei\CACViT\x_y.pth")
# # Create a dummy input tensor with shape (B, N, C)
# # x = torch.randn(16, 624, 768)  # B=16, N=624, C=3
#
# # Forward pass
# x_restored, y_restored = module(x)
#
# # Print shapes of outputs
# print(x_restored.shape)  # Should be (16, 3, 24, 24)
# print(y_restored.shape)  # Should be (3, 16, 16, 3)
