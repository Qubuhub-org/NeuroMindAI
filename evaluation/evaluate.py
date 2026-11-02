"""Evaluate a checkpoint using average next-token cross-entropy and perplexity."""

import argparse
import math

import torch
from torch.utils.data import DataLoader

from datasets.text import TextWindowDataset
from neuromind.inference import load_model


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        _, loss = model(inputs, targets)
        if loss is None:
            raise RuntimeError("evaluation loss was not computed")
        count = targets.numel()
        total_loss += loss.item() * count
        total_tokens += count
    if total_tokens == 0:
        raise ValueError("evaluation dataset is empty")
    mean_loss = total_loss / total_tokens
    return {"cross_entropy": mean_loss, "perplexity": math.exp(mean_loss)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a NeuroMindAI checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, help="UTF-8 evaluation corpus")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    model, device = load_model(args.checkpoint, args.device)
    dataset = TextWindowDataset(args.data, model.config.context_length)
    loader = DataLoader(dataset, batch_size=args.batch_size)
    metrics = evaluate(model, loader, device)
    print(f"cross_entropy={metrics['cross_entropy']:.6f}")
    print(f"perplexity={metrics['perplexity']:.4f}")


if __name__ == "__main__":
    main()
