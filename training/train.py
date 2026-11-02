"""Training engine with streaming input, DDP, CUDA AMP, and resumable checkpoints."""

import argparse
import os
import random
from pathlib import Path

import torch
from torch import distributed
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from datasets.streaming import StreamingTextWindowDataset
from datasets.text import TextWindowDataset
from neuromind import ModelConfig, NeuroMindLM
from tokenizer import BpeTokenizer


def _save_checkpoint(
    output: Path,
    model: NeuroMindLM,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    tokenizer: BpeTokenizer,
    step: int,
    seed: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 2,
            "config": model.config.to_dict(),
            "tokenizer": tokenizer.to_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "training": {"steps": step, "seed": seed},
        },
        output,
    )


def train(args: argparse.Namespace) -> Path:
    if args.steps < 1 or args.batch_size < 1 or args.grad_accum_steps < 1:
        raise ValueError("steps, batch_size, and grad_accum_steps must be positive")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    process_group_started = False

    if is_distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        distributed.init_process_group(backend=backend)
        process_group_started = True

    try:
        if args.device:
            device = torch.device(args.device)
            if device.type == "cuda" and device.index is None:
                device = torch.device(f"cuda:{local_rank}" if is_distributed else "cuda")
        elif torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}" if is_distributed else "cuda")
        else:
            device = torch.device("cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)

        random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + rank)

        use_amp = bool(args.amp)
        if use_amp and device.type != "cuda":
            raise ValueError("--amp requires a CUDA device")
        if args.streaming and is_distributed:
            raise ValueError(
                "streaming DDP is not enabled yet; use map-style data with torchrun"
            )
        amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
        use_scaler = use_amp and amp_dtype == torch.float16
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

        corpus = Path(args.data).read_text(encoding="utf-8")
        tokenizer = BpeTokenizer.train(
            corpus, vocab_size=args.vocab_size, min_frequency=args.min_frequency
        )
        config = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            context_length=args.context_length,
            d_model=args.d_model,
            n_layers=args.layers,
            n_heads=args.heads,
            d_ff=args.d_ff,
            dropout=args.dropout,
        )

        if args.streaming:
            dataset = StreamingTextWindowDataset(
                args.data, config.context_length, tokenizer
            )
            sampler = None
        else:
            dataset = TextWindowDataset(args.data, config.context_length, tokenizer)
            sampler = (
                DistributedSampler(
                    dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True,
                    seed=args.seed,
                )
                if is_distributed
                else None
            )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=not args.streaming and sampler is None,
            sampler=sampler,
            drop_last=False,
            generator=torch.Generator().manual_seed(args.seed + rank),
        )
        iterator = iter(loader)
        raw_model = NeuroMindLM(config).to(device)
        optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.learning_rate)
        step = 0

        if args.resume:
            checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
            if checkpoint.get("config") != config.to_dict():
                raise ValueError("resume checkpoint model configuration does not match")
            if checkpoint.get("tokenizer") != tokenizer.to_dict():
                raise ValueError(
                    "resume checkpoint tokenizer does not match current training corpus"
                )
            raw_model.load_state_dict(checkpoint["model_state"])
            if "optimizer_state" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            if "scaler_state" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state"])
            step = int(checkpoint.get("training", {}).get("steps", 0))
            if step >= args.steps:
                raise ValueError(
                    "resume checkpoint has already reached the requested step count"
                )

        model = (
            DistributedDataParallel(
                raw_model,
                device_ids=[device.index] if device.type == "cuda" else None,
            )
            if is_distributed
            else raw_model
        )
        epoch = 0
        model.train()

        while step < args.steps:
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                epoch += 1
                if sampler is not None:
                    sampler.set_epoch(epoch)
                iterator = iter(loader)
                inputs, targets = next(iterator)

            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            batch_inputs = [(inputs, targets)]
            for _ in range(args.grad_accum_steps - 1):
                try:
                    batch_inputs.append(next(iterator))
                except StopIteration:
                    epoch += 1
                    if sampler is not None:
                        sampler.set_epoch(epoch)
                    iterator = iter(loader)
                    batch_inputs.append(next(iterator))

            for inputs, targets in batch_inputs:
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=use_amp
                ):
                    _, loss = model(inputs, targets)
                    if loss is None:
                        raise RuntimeError("training loss was not computed")
                    scaled_loss = loss / args.grad_accum_steps
                scaler.scale(scaled_loss).backward()
                accumulated_loss += loss.detach().float().item()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            step += 1

            mean_loss = accumulated_loss / args.grad_accum_steps
            if rank == 0 and (
                step == 1 or step % args.log_every == 0 or step == args.steps
            ):
                print(f"step={step} loss={mean_loss:.4f}")

            if rank == 0 and args.save_every > 0 and step % args.save_every == 0:
                _save_checkpoint(
                    Path(args.output),
                    raw_model,
                    optimizer,
                    scaler,
                    tokenizer,
                    step,
                    args.seed,
                )

        output = Path(args.output)
        if rank == 0:
            _save_checkpoint(
                output, raw_model, optimizer, scaler, tokenizer, step, args.seed
            )
            print(f"tokenizer vocabulary: {tokenizer.vocab_size}")
            print(f"checkpoint saved: {output}")
        if is_distributed:
            distributed.barrier()
        return output
    finally:
        if process_group_started:
            distributed.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the NeuroMindAI language model.")
    parser.add_argument("--data", required=True, help="UTF-8 plain-text training corpus")
    parser.add_argument("--output", default="checkpoints/neuromind.pt")
    parser.add_argument("--resume", default=None, help="resume from a compatible checkpoint")
    parser.add_argument("--steps", type=int, default=1000, help="total optimizer update steps")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--min-frequency", type=int, default=2)
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
    parser.add_argument("--streaming", action="store_true", help="read corpus incrementally")
    parser.add_argument("--amp", action="store_true", help="enable CUDA automatic mixed precision")
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16"), default="bf16")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
