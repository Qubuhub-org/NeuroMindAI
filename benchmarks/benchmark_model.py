"""Measure forward/backward token throughput for a configured NeuroMind model."""

import argparse
import json
import time

import torch

from neuromind import ModelConfig, NeuroMindLM


def benchmark(
    config: ModelConfig,
    batch_size: int,
    sequence_length: int,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> dict[str, float | int | str]:
    if batch_size < 1 or sequence_length < 1 or sequence_length > config.context_length:
        raise ValueError("batch size and sequence length must be positive and within context")
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be non-negative and iterations must be positive")

    model = NeuroMindLM(config).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    inputs = torch.randint(
        0, config.vocab_size, (batch_size, sequence_length), device=device
    )
    targets = torch.randint(
        0, config.vocab_size, (batch_size, sequence_length), device=device
    )

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets)
        if loss is None:
            raise RuntimeError("benchmark loss was not computed")
        loss.backward()
        optimizer.step()

    for _ in range(warmup):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    for _ in range(iterations):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    token_count = batch_size * sequence_length * iterations
    result: dict[str, float | int | str] = {
        "device": str(device),
        "architecture": config.architecture,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "iterations": iterations,
        "elapsed_seconds": elapsed,
        "tokens_per_second": token_count / elapsed,
    }
    if device.type == "cuda":
        result["peak_memory_bytes"] = torch.cuda.max_memory_allocated(device)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark NeuroMind model throughput.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=1024)
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    config = ModelConfig(
        vocab_size=args.vocab_size,
        context_length=max(args.sequence_length, 1),
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        d_ff=args.d_ff,
    )
    print(
        json.dumps(
            benchmark(
                config,
                args.batch_size,
                args.sequence_length,
                args.warmup,
                args.iterations,
                device,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
