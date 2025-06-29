# soft_dtw.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftDTW(nn.Module):
    def __init__(self, gamma=1.0, normalize=False):
        super(SoftDTW, self).__init__()
        self.gamma = gamma
        self.normalize = normalize

    def forward(self, x, y):
        B, T, D = x.shape
        D_xy = torch.norm(x.unsqueeze(2) - y.unsqueeze(1), dim=3, p=2) ** 2

        R = torch.zeros((B, T + 2, T + 2), device=x.device) + float('inf')
        R[:, 0, 0] = 0

        for i in range(1, T + 1):
            for j in range(1, T + 1):
                r0 = -R[:, i - 1, j - 1] / self.gamma
                r1 = -R[:, i - 1, j] / self.gamma
                r2 = -R[:, i, j - 1] / self.gamma
                rmax = torch.max(torch.max(r0, r1), r2)
                rsum = torch.exp(r0 - rmax) + torch.exp(r1 - rmax) + torch.exp(r2 - rmax)
                softmin = -self.gamma * (torch.log(rsum) + rmax)
                R[:, i, j] = D_xy[:, i - 1, j - 1] + softmin

        if self.normalize:
            return R[:, T, T] / T
        else:
            return R[:, T, T]
