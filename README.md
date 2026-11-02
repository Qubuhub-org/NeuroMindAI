# NeuroMindAI 🧠

NeuroMindAI is a from-scratch decoder-only Transformer research project.

## v0.1

This milestone establishes an independent model architecture and training path rather than renaming an existing checkpoint.

Core components:

- RMSNorm
- RoPE positional encoding
- causal self-attention
- configurable GQA-style key/value heads
- gated SiLU MLP
- residual Transformer blocks
- tied input/output embeddings
- next-token cross-entropy training
- autoregressive generation
- PyTorch checkpoint export

## Architecture

```text
tokens
  │
  ▼
Embedding
  │
  ▼
┌──────────────────────┐
│ NeuroMindBlock × N   │
│  RMSNorm             │
│  RoPE + Attention    │
│  Residual            │
│  RMSNorm             │
│  Gated SiLU MLP      │
│  Residual            │
└──────────┬───────────┘
           ▼
       Final RMSNorm
           ▼
         LM Head
           ▼
        Logits
```

## Run

```bash
python -m pip install -e .
python -m training.train --steps 100
pytest -q
```

The training script uses a tiny deterministic synthetic sequence so the complete forward/backward/checkpoint path can be validated without downloading a dataset.

## Roadmap

1. tokenizer and dataset pipeline
2. real pretraining corpus
3. mixed precision and distributed training
4. efficient KV-cache generation
5. checkpoint conversion
6. GGUF/GGML C++ inference runtime
7. evaluation and reproducibility tooling
8. multimodal and memory systems above the language core

**Status:** experimental v0.1 — architecture/runtime scaffold, not a pretrained language model.
