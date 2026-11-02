# NeuroMindAI

**An independent, research-oriented language-model foundation.**

NeuroMindAI provides a native decoder-only architecture, byte-level BPE tokenizer, training engine, evaluation, and checkpoint-based inference.

> **Status:** research foundation. No pretrained weights are included. Useful language capability requires training on a large, curated, appropriately licensed dataset.

## Included

- Native byte-level BPE tokenizer with deterministic merges and full byte fallback.
- Versioned tokenizer state embedded in model checkpoints.
- **NeuroMind v1:** RMSNorm, rotary position embeddings, causal self-attention, SwiGLU MLP, and residual decoder blocks.
- Legacy Transformer checkpoint compatibility.
- Sliding-window next-token dataset utilities.
- Training with gradient accumulation, optional CUDA mixed precision, gradient clipping, periodic checkpoints, and resume.
- Checkpoint-based generation and cross-entropy/perplexity evaluation.
- Unit tests and GitHub Actions CI.

## Repository layout

```text
neuromind/       model configuration, native architecture, inference
tokenizer/       byte fallback and native BPE implementation
training/        training CLI and checkpoint lifecycle
datasets/        sliding-window text data
evaluation/      cross-entropy and perplexity evaluation
configs/         model defaults
docs/            architecture and development notes
tests/           automated tests
examples/        small demonstration corpus
```

## Requirements and installation

- Python 3.10+
- PyTorch 2.2+

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

On Windows PowerShell, activate with `.venv\\Scripts\\Activate.ps1`. For CUDA, install the PyTorch wheel matching your CUDA runtime using the official PyTorch installation selector.

## Train

```bash
python -m training.train \
  --data examples/tiny_corpus.txt \
  --steps 100 \
  --batch-size 8 \
  --vocab-size 320 \
  --grad-accum-steps 2
```

For CUDA mixed precision, add `--amp` and choose `--amp-dtype bf16` or `--amp-dtype fp16`. Resume a compatible run with `--resume checkpoints/neuromind.pt`. The demonstration corpus is only for smoke tests; it is not sufficient to train a useful language model.

## Generate

```bash
python -m neuromind.inference \
  --checkpoint checkpoints/neuromind.pt \
  --prompt "NeuroMindAI"
```

## Evaluate

```bash
python -m evaluation.evaluate \
  --checkpoint checkpoints/neuromind.pt \
  --data path/to/held-out.txt
```

Perplexity comparisons require the same tokenizer and compatible evaluation data.

## Model defaults

| Setting | Default |
|---|---:|
| BPE vocabulary target | 512 |
| Context length | 256 in config; 128 in training CLI |
| Hidden width | 256 in config; 128 in training CLI |
| Decoder layers | 4 |
| Attention heads | 4 |
| Feed-forward width | 1,024 in config; 512 in training CLI |
| Dropout | 0.1 |
| RoPE theta | 10,000 |

## Engineering principles

1. Security before features.
2. Simplicity before complexity.
3. Working code before optimization.
4. Tests before deployment.
5. Documentation before release.

## Limitations and roadmap

This is not a pretrained or production foundation model. The BPE trainer currently uses repeated in-memory scans and needs optimization before large-scale corpus preparation. Distributed training, streaming data, tokenizer throughput benchmarks, instruction tuning, tool use, and safety evaluation remain future work.

See [Architecture](docs/ARCHITECTURE.md) and [Development](docs/DEVELOPMENT.md).

## License

No license has been selected. Do not assume the repository is open-source licensed until the owner adds and approves a license file.
