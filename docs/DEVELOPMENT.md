# Development

## Local checks

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

## Smoke training

```bash
python -m training.train --data examples/tiny_corpus.txt --steps 5 --batch-size 2 --context-length 32 --d-model 32 --layers 1 --heads 4 --d-ff 64 --dropout 0
```

## Experiment hygiene

- Keep datasets outside version control unless their license and size are explicitly approved.
- Record dataset provenance, license, preprocessing, random seed, and configuration.
- Never commit API keys, credentials, private data, or unreviewed model checkpoints.
- Validate checkpoints before deployment and do not load untrusted pickle-based artifacts.
- Report hardware, precision, throughput, training steps, and evaluation results with every benchmark.
- Keep the stable branch protected; develop changes on feature branches and merge through reviewed pull requests.
