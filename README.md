# NeuroMindAI

**An independent, research-oriented language-model foundation.**

NeuroMindAI starts with a compact, decoder-only Transformer implemented in native PyTorch modules and a reversible UTF-8 byte tokenizer. The repository is intentionally structured so tokenization, model architecture, data, training, evaluation, and inference can evolve independently.

> **Status:** foundational implementation. No pretrained weights are included. A model must be trained on an appropriately licensed dataset before it can be expected to produce useful language.

## What is included

- A byte-level tokenizer with deterministic, lossless UTF-8 round-tripping.
- A causal Transformer language model with token and positional embeddings.
- Sliding-window text dataset utilities.
- A reproducible, configurable training entry point.
- Checkpoint-based text generation.
- Unit tests and a GitHub Actions test workflow.

## Repository layout

```text
neuromind/
  __init__.py
  config.py
  model.py
  inference.py
tokenizer/
  __init__.py
  byte_tokenizer.py
training/
  __init__.py
  train.py
datasets/
  __init__.py
  text.py
configs/
  base.json
docs/
  ARCHITECTURE.md
  DEVELOPMENT.md
tests/
  test_tokenizer.py
  test_model.py
examples/
  tiny_corpus.txt
.github/workflows/
  tests.yml
```

## Requirements

- Python 3.10+
- PyTorch 2.2+
- CPU is supported; CUDA is used when available and requested by the runtime.

Install:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

For a CUDA-enabled PyTorch build, install the matching wheel from the official PyTorch installation selector before installing this project with `--no-deps`.

## Train a small model

Start with the included demonstration corpus:

```bash
python -m training.train --data examples/tiny_corpus.txt --steps 100 --batch-size 8
```

This is a smoke-test configuration, not a meaningful pretraining run. Use a large, curated, licensed corpus and tune the model and optimization settings for real experiments.

## Generate text

```bash
python -m neuromind.inference --checkpoint checkpoints/neuromind.pt --prompt "NeuroMindAI"
```

The checkpoint must first be produced by training. Generation supports temperature and top-k sampling.

## Configuration

The default architecture is deliberately small:

| Setting | Default |
|---|---:|
| Vocabulary | 256 byte values |
| Context length | 256 tokens |
| Hidden width | 256 |
| Transformer layers | 4 |
| Attention heads | 4 |
| Feed-forward width | 1,024 |
| Dropout | 0.1 |

Edit `configs/base.json` or pass corresponding command-line options to the training script.

## Engineering principles

1. Security before features.
2. Simplicity before complexity.
3. Working code before optimization.
4. Tests before deployment.
5. Documentation before release.

## Limitations and roadmap

This initial implementation is a research baseline, not a production foundation model. It does not include pretrained weights, distributed training, mixed-precision orchestration, a subword/BPE tokenizer, instruction tuning, tool use, or a safety-alignment layer. Those capabilities should be added behind tested interfaces and justified by measurable requirements.

See [Architecture](docs/ARCHITECTURE.md) and [Development](docs/DEVELOPMENT.md).

## License

No license has been selected in this initialization. Do not assume the code is open-source licensed until the repository owner adds and approves a license file.
