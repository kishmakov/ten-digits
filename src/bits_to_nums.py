import torch
import torch.nn as nn


class BitsToNumsNet(nn.Module):
    def __init__(self, bits: int, nums: int, hidden: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(bits, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, nums)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
