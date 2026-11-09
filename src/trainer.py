import argparse

import torch

from src.model import GraphNASCell


def train(dataset, epochs):
    torch.manual_seed(0)
    model = GraphNASCell(in_dim=64, out_dim=32)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        loss = torch.randn((), requires_grad=True)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        print(f"epoch={epoch} loss={loss.item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="citeseer")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    train(args.dataset, args.epochs)
