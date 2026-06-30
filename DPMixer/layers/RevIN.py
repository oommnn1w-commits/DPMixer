# layers/RevIN.py
import torch
import torch.nn as nn


class RevIN(nn.Module):
    #处理非平稳性
    def __init__(self, num_features, eps=1e-5, affine=True, subtract_last=False):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.affine_weight = None
            self.affine_bias = None

    def forward(self, x, mode):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        # 计算每个特征的均值和标准差
        # x: [B, N, L]
        self.mean = torch.mean(x, dim=2, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=2, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        # 归一化操作
        # x: [B, N, L]
        x = x - self.mean
        x = x / self.stdev

        # 应用仿射变换
        if self.affine:
            # 确保affine_weight维度正确 [1, N, 1]
            affine_weight = self.affine_weight.unsqueeze(0).unsqueeze(2)
            affine_bias = self.affine_bias.unsqueeze(0).unsqueeze(2)
            x = x * affine_weight + affine_bias
        return x

    def _denormalize(self, x):
        # 反归一化操作
        # x: [B, N, L]
        if self.affine:
            # 确保affine_weight维度正确 [1, N, 1]
            affine_weight = self.affine_weight.unsqueeze(0).unsqueeze(2)
            affine_bias = self.affine_bias.unsqueeze(0).unsqueeze(2)
            x = (x - affine_bias) / (affine_weight + self.eps)

        x = x * self.stdev
        x = x + self.mean
        return x