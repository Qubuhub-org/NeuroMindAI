# Architecture

## End-to-end data path

`UTF-8 corpus → BpeTokenizer → token windows → NeuroMindLM v1 → next-token loss`

The BPE tokenizer is learned from the training corpus. Its exact merge table is embedded in each checkpoint; inference and evaluation reconstruct the tokenizer from the checkpoint.

## Native byte-level BPE tokenizer

The vocabulary begins with all 256 possible byte values. Training repeatedly counts adjacent token pairs, selects the most frequent pair (with deterministic lexicographic tie-breaking), and appends a new token representing that pair. Training stops at the requested vocabulary size or when no pair meets the minimum frequency.

The byte fallback supports unseen UTF-8 content without an unknown-token failure. The merge list is versioned and serialized as integer pairs. The current trainer performs repeated in-memory scans and is intended for small experiments, not billion-token preprocessing.

The legacy `ByteTokenizer` remains supported for checkpoints created before tokenizer metadata was embedded.

## NeuroMind v1 decoder

NeuroMind v1 is a native pre-normalized decoder architecture implemented from explicit PyTorch modules:

1. **Token embedding** maps token IDs to the hidden width.
2. **RMSNorm** normalizes activations without mean subtraction.
3. **Fused causal self-attention** projects Q/K/V, applies rotary position embeddings (RoPE), and uses PyTorch scaled dot-product attention with a causal constraint.
4. **SwiGLU feed-forward network** applies a SiLU-gated projection and down projection.
5. **Residual connections** wrap attention and feed-forward sublayers.
6. **Final RMSNorm and tied output projection** produce next-token logits.

The head dimension must be even for rotary embeddings. The architecture uses no learned absolute position embedding. Attention is causal in both training and inference. The model vocabulary size is derived from the trained tokenizer to prevent embedding/token-ID mismatches.

## Legacy checkpoint compatibility

The original baseline used `nn.TransformerEncoderLayer` blocks and learned positional embeddings. The inference loader recognizes those checkpoint keys and selects the legacy architecture so existing baseline checkpoints remain loadable.

## Training and checkpoints

The training engine supports gradient accumulation, optional CUDA automatic mixed precision (FP16 or BF16), gradient clipping, periodic checkpoint writes, and resume from a compatible checkpoint. A checkpoint includes model configuration, model weights, tokenizer merges, optimizer/scaler state, and basic training metadata.

Resume requires the same model configuration and tokenizer state. This protects against accidentally continuing a run with a changed vocabulary.

## Boundaries

This is a compact research implementation, not a pretrained foundation model. Distributed training, streaming datasets, optimized large-corpus BPE, instruction tuning, tool use, and safety evaluation remain future work.


## Incremental inference and KV cache

NeuroMind v1 exposes `forward_with_cache`, which stores per-layer rotated keys and values. Subsequent tokens reuse those tensors instead of recomputing attention projections for the full prefix. Cached attention uses an explicit offset-aware causal mask; this is important because single-token decoding with a naïve triangular mask can expose incorrect positions. Generation uses the cache and rebuilds it when the context window rolls over. Legacy Transformer checkpoints continue through the uncached path.

## Streaming and distributed data

`StreamingTextWindowDataset` reads UTF-8 lines incrementally and emits overlapping next-token windows while retaining a context-sized carry between lines. The training CLI exposes this with `--streaming`. The current tokenizer-training stage still reads the corpus into memory, so this is streaming batch ingestion rather than a fully streaming end-to-end preprocessing pipeline.

Map-style training supports PyTorch DistributedDataParallel under `torchrun`, using a `DistributedSampler` to shard samples and synchronize gradients. Streaming DDP is intentionally rejected for now because line-based shards can have unequal batch counts and cause synchronization deadlocks. This limitation is explicit rather than silently risking a hung multi-GPU job.

## Performance measurement

`python -m benchmarks.benchmark_model` measures synthetic forward/backward throughput, reports parameter count and tokens per second, and records peak allocated CUDA memory when available. It is a local throughput measurement, not a quality benchmark. Language quality must be evaluated separately on held-out datasets.
