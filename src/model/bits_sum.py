import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from src.model.base_model import BaseModel


@dataclass
class BitsSumConfig:
    bits: int
    hidden: int = 2


class BitsSumNet(BaseModel):
    def __init__(self, config: BitsSumConfig):
        super().__init__()
        self._config = config
        self.fc1 = nn.Linear(2 * config.bits, config.hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(config.hidden, config.bits + 1)

    @property
    def config(self) -> BitsSumConfig:
        return self._config

    def init(self, seed):
        torch.manual_seed(seed)

        for layer in [self.fc1, self.fc2]:
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            nn.init.normal_(layer.bias, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

    def get_train_batch(self):
        X = []
        Y = []

        num_values = 1 << self.config.bits
        for i in range(num_values):
            for j in range(num_values):
                bits_i = _get_bits_vector(i, self.config.bits) # bits
                bits_j = _get_bits_vector(j, self.config.bits) # bits

                X.append(torch.cat([bits_i, bits_j])) # (2 * bits)
                Y.append(_get_bits_vector(i + j, self.config.bits + 1)) # bits+1

        return torch.stack(X), torch.stack(Y)

    def get_test_batch(self):
        return self.get_train_batch()

    @torch.no_grad()
    def get_test_metrics(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        logits = self.forward(x)
        probs = torch.sigmoid(logits) # probabilities for each bit [0, 1]
        preds = (probs > 0.5).float() # threshold at 0.5 to get binary predictions
        bit_acc = (preds == y).float().mean().item() # bit-level accuracy: percentage
        full_match = (preds == y).all(dim=-1).float().mean().item() # percentage of samples where all bits are correct

        return {
            "rmse": torch.sqrt(torch.mean((probs - y) ** 2)).item(),
            "bit_acc": bit_acc,
            "acc": full_match,
        }

    def get_name(self) -> str:
        return f"B{self.config.bits}SumH{self.config.hidden}"


def _get_bits_vector(id: int, bits: int) -> torch.Tensor:
    vec = torch.zeros(bits)
    for b in range(bits):
        vec[b] = (id >> b) & 1

    return vec

