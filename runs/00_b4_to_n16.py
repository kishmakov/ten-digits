import pathlib
import sys
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bits_to_nums import BitsToNumsNet


def main() -> None:
    model = BitsToNumsNet(4, 16)

    x = torch.randn(2, 4)
    y = model(x)
    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(y.shape))
    print("First output row:", y[0].tolist())


if __name__ == "__main__":
    main()
