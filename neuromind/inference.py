"""Checkpoint loading and autoregressive text generation."""

import argparse
from pathlib import Path

import torch

from neuromind import ModelConfig, NeuroMindLM
from tokenizer import BpeTokenizer, ByteTokenizer


def load_model(
    checkpoint: str | Path, device: str | None = None
) -> tuple[NeuroMindLM, torch.device, BpeTokenizer | ByteTokenizer]:
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = torch.load(checkpoint, map_location=target, weights_only=True)
    config_values = dict(payload["config"])
    if "position_embedding.weight" in payload["model_state"]:
        config_values["architecture"] = "legacy_transformer"
    config = ModelConfig.from_dict(config_values)
    tokenizer_state = payload.get("tokenizer")
    if tokenizer_state is None:
        tokenizer = ByteTokenizer()
    elif tokenizer_state.get("type") == "bpe":
        tokenizer = BpeTokenizer.from_dict(tokenizer_state)
    else:
        raise ValueError("checkpoint contains an unsupported tokenizer")
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError("checkpoint tokenizer vocabulary does not match model configuration")
    model = NeuroMindLM(config).to(target)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, target, tokenizer


@torch.no_grad()
def generate(
    model: NeuroMindLM,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int | None = 40,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    if input_ids.ndim != 2 or input_ids.size(1) == 0:
        raise ValueError("input_ids must be a non-empty [batch, sequence] tensor")
    model.eval()
    for _ in range(max_new_tokens):
        context = input_ids[:, -model.config.context_length :]
        logits, _ = model(context)
        next_logits = logits[:, -1, :] / temperature
        if top_k is not None:
            if top_k < 1:
                raise ValueError("top_k must be >= 1 or None")
            k = min(top_k, next_logits.size(-1))
            cutoff = torch.topk(next_logits, k, dim=-1).values[:, -1, None]
            next_logits = next_logits.masked_fill(next_logits < cutoff, float("-inf"))
        probabilities = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        input_ids = torch.cat((input_ids, next_token), dim=1)
    return input_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a NeuroMindAI checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    model, device, tokenizer = load_model(args.checkpoint, args.device)
    encoded = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    output = generate(model, encoded, args.max_new_tokens, args.temperature, args.top_k)
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
