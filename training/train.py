import argparse
import torch
from neuromind import NeuroMindConfig, NeuroMindForCausalLM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = NeuroMindConfig(
        vocab_size=256, hidden_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_kv_heads=4,
        intermediate_size=512, max_position_embeddings=128,
    )
    model = NeuroMindForCausalLM(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    seq = torch.arange(128, device=args.device).unsqueeze(0) % config.vocab_size
    for step in range(args.steps):
        out = model(seq[:, :-1], labels=seq[:, 1:])
        optimizer.zero_grad(set_to_none=True)
        out["loss"].backward()
        optimizer.step()
        if step % 10 == 0:
            print(f"step={step:04d} loss={out['loss'].item():.4f}")

    torch.save({"config": vars(config), "state_dict": model.state_dict()}, "neuromindai-v0.1.pt")

if __name__ == "__main__":
    main()
