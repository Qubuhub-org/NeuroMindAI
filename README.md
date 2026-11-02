# NeuroMindAI

**An independent, research-oriented language-model foundation.**

NeuroMindAI currently provides a compact decoder-only Transformer, a native byte-level BPE tokenizer with full byte fallback, a training pipeline, evaluation, and checkpoint-based inference.

> **Status:** research foundation. No pretrained weights are included. Useful language capability requires training on a large, curated, appropriately licensed dataset.

## Included

- Native BPE tokenizer trained from UTF-8 text, with deterministic merges and reversible byte fallback.
- Versioned tokenizer serialization embedded in model checkpoints.
- Causal Transformer language model with token and positional embeddings.
- Sliding-window next-token dataset utilities.
- Reproducible training entry point with gradient clipping.
- Checkpoint-based text generation and evaluation (cross-entropy/perplexity).
- Unit tests and GitHub Actions CI.

## Repository layout

```text
neuromind/       model configuration, architecture, inference
tokenizer/       byte fallback and native BPE implementation
training/        training CLI
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

## Train a small model

```bash
python -m training.train \
  --data examples/tiny_corpus.txt \
  --steps 100 \
  --batch-size 8 \
  --vocab-size 320
```

The tokenizer is learned from the supplied training corpus. Its vocabulary size becomes the model's vocabulary size, and its merge table is stored in the checkpoint. The demonstration corpus is only for pipeline smoke tests; it is not sufficient to train a useful language model.

## Generate text

```bash
python -m neuromind.inference \
  --checkpoint checkpoints/neuromind.pt \
  --prompt "NeuroMindAI"
```

## Evaluate

Use a held-out UTF-8 corpus that was not used for training:

```bash
python -m evaluation.evaluate \
  --checkpoint checkpoints/neuromind.pt \
  --data path/to/held-out.txt
```

The reported perplexity is meaningful only when comparing models evaluated with the same tokenizer and compatible evaluation data.

## Baseline model defaults

| Setting | Default |
|---|---:|
| BPE vocabulary target | 512 |
| Context length | 256 in config; 128 in training CLI |
| Hidden width | 256 in config; 128 in training CLI |
| Transformer layers | 4 |
| Attention heads | 4 |
| Feed-forward width | 1,024 in config; 512 in training CLI |
| Dropout | 0.1 |

## Engineering principles

1. Security before features.
2. Simplicity before complexity.
3. Working code before optimization.
4. Tests before deployment.
5. Documentation before release.

## Limitations and roadmap

This is not a pretrained or production foundation model. The current BPE trainer uses repeated in-memory scans and needs optimization before large-scale corpus preparation. Distributed training, mixed precision, data streaming, tokenizer throughput benchmarks, instruction tuning, tool use, and safety evaluation remain future work.

See [Architecture](docs/ARCHITECTURE.md) and [Development](docs/DEVELOPMENT.md).

## License

No license has been selected. Do not assume the repository is open-source licensed until the owner adds and approves a license file.
