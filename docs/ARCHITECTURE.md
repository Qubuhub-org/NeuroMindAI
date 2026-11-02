# Architecture

## Data path

`UTF-8 corpus → ByteTokenizer → token windows → NeuroMindLM → next-token loss`

Inference reverses the path: prompt text is encoded, tokens are sampled autoregressively, and the resulting byte sequence is decoded.

## Tokenizer

The initial tokenizer maps each UTF-8 byte to its unsigned integer value (0–255). It is deterministic, has no training phase, and preserves valid UTF-8 text exactly. This avoids a hidden dependency on a third-party vocabulary. Its trade-off is sequence length: byte tokens are less compact than subword tokens. A future BPE or unigram tokenizer should be versioned and checkpointed so model weights cannot silently use a different token mapping.

## Model

The model is a decoder-only Transformer:
1. Token embedding and learned positional embedding.
2. A stack of pre-normalized Transformer encoder blocks with a boolean causal mask.
3. Final layer normalization.
4. Vocabulary projection tied to the token embedding matrix.

For a batch of token sequences, logits have shape `[batch, sequence, vocabulary]`. Training compares each position's logits with the corresponding next-token target.

## Checkpoints

A checkpoint contains a plain configuration dictionary and a model state dictionary. The inference loader uses PyTorch's `weights_only=True` loading mode. Treat checkpoints as trusted artifacts nonetheless and record their provenance, dataset, and training parameters for research runs.

## Deliberate boundaries

This baseline does not claim to provide a pretrained foundation model, distributed training, a production tokenizer, model-serving API, alignment, or domain-specific capability. Add those as independently tested modules rather than coupling them into the core model.
