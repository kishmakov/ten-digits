import pathlib
import sys
import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model.bits_sum import BitsSumNet, BitsSumConfig
from src.training import TrainingConfig, Training


if __name__ == "__main__":
    model_config = BitsSumConfig(bits=6, hidden=15)
    training_config = TrainingConfig(epochs=50000, device="cuda")

    model = BitsSumNet(model_config)
    training = Training(training_config, model)

    training.train(torch.nn.BCEWithLogitsLoss)

