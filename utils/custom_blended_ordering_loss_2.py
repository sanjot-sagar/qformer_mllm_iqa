import torch
import torchvision.transforms as transforms
from torch import nn
from random import randint
import torch.nn.functional as F


class CustomBlendedOrderingLoss(nn.Module):
    def __init__(self):
        super(CustomBlendedOrderingLoss, self).__init__()

    def forward(self, x):  # x => bbs x bl
        # append 1 and 0
        x = torch.cat((torch.ones(x.size(0), 1, device=x.device), x, torch.zeros(
            x.size(0), 1, device=x.device)), dim=-1)   # x => B x (D+2)
        window_size = 3
        c = torch.cat([
            x.unfold(1, window_size, 1),
            torch.flip(x, dims=[1]).unfold(1, window_size, 1)
        ], dim=1)
        # c => 16 image pairs * 6 triplets * 3
        # Calculate differences and squares using vectorized operations on the GPU
        diff1 = (c[:, 0] - c[:, 1])**2
        diff2 = (c[:, 0] - c[:, 2])**2

        # Simplify the computation using PyTorch element-wise operations on the GPU
        result = (torch.relu(diff1 - diff2 + 0.5 * (diff1 + diff2))).mean()

        return result
