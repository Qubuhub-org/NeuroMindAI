"""Minimal, reproducible next-token training loop."""

import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.text import TextWindowDataset
from neuromind import ModelConfig, NeuroMindLM


def train(args: argparse.Namespace) -> Path:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config = ModelConfig(
        context_length=args.context_length,
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
    )
    dataset = TextWindowDataset(args.data, config.context_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    model = NeuroMindLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model.train()

    step = 0
    while step < args.steps:
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            _, loss = model(inputs, targets)
            if loss is None:
                raise RuntimeError("training loss was not computed")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            step += 1
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                print(f"step={step} loss={loss.item():.4f}")
            if step >= args.steps:
                break

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config.to_dict(), "model_state": model.state_dict()}, output)
    print(f"checkpoint saved: {output}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the NeuroMindAI language model.")
    parser.add_argument("--data", required=True, help="UTF-8 plain-text training corpus")
    parser.add_argument("--output", default="checkpoints/neuromind.pt")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
