# Architecture

## Data path

`UTF-8 corpus → BpeTokenizer → token windows → NeuroMindLM → next-token loss`

The tokenizer is trained on the training corpus and its exact merge table is embedded in each checkpoint. Inference and evaluation reconstruct that tokenizer from the checkpoint rather than retraining or guessing a vocabulary.

## Native byte-level BPE tokenizer

The vocabulary begins with all 256 possible byte values. Training repeatedly counts adjacent token pairs, selects the most frequent pair (with deterministic lexicographic tie-breaking), and appends a new token representing that pair. The process stops at the requested vocabulary size or when no pair meets the minimum frequency.

This gives the tokenizer complete byte fallback for unseen text and lossless round-tripping of valid UTF-8. The merge list is versioned and serialized as integer pairs. The initial implementation favors clarity over large-corpus throughput: it performs repeated scans of the training sequence and is intended for tokenizer experiments, not billion-token preprocessing jobs.

The legacy `ByteTokenizer` remains supported for older checkpoints without embedded tokenizer metadata.

## Model

The current model is a decoder-only Transformer:
1. Token embedding and learned positional embedding.
2. A stack of pre-normalized Transformer encoder blocks with a boolean causal mask.
3. Final layer normalization.
4. Vocabulary projection tied to the token embedding matrix.

For a batch of token sequences, logits have shape `[batch, sequence, vocabulary]`. Training compares each position's logits with the corresponding next-token target. The model vocabulary size is derived from the trained tokenizer to prevent embedding/token-ID mismatches.

## Checkpoints

New checkpoints contain a format version, model configuration, model state dictionary, complete tokenizer state, and basic training metadata. The inference loader uses PyTorch's `weights_only=True` loading mode. Treat checkpoints as trusted artifacts nonetheless and record dataset provenance and training parameters for research runs.

## Deliberate boundaries

This is still a compact research baseline, not a pretrained foundation model. It does not claim distributed training, optimized large-corpus tokenization, instruction tuning, tool use, or a safety-alignment layer. Add those as independently tested modules rather than coupling them into the core model.
