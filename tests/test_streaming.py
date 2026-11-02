import torch

from datasets.streaming import StreamingTextWindowDataset
from tokenizer import BpeTokenizer


def test_streaming_dataset_emits_contiguous_windows(tmp_path):
    corpus = "abcdefghij\nklmnopqrst\n"
    path = tmp_path / "corpus.txt"
    path.write_text(corpus, encoding="utf-8")
    tokenizer = BpeTokenizer.train(corpus, vocab_size=256)
    dataset = StreamingTextWindowDataset(path, 4, tokenizer)
    samples = list(dataset)
    assert samples
    for inputs, targets in samples:
        assert inputs.shape == (4,)
        assert targets.shape == (4,)
        assert torch.equal(inputs[1:], targets[:-1])


def test_streaming_dataset_shards_lines_by_rank(tmp_path):
    corpus = "line one has tokens\nline two has tokens\nline three has tokens\n"
    path = tmp_path / "corpus.txt"
    path.write_text(corpus, encoding="utf-8")
    tokenizer = BpeTokenizer.train(corpus, vocab_size=256)
    rank_zero = list(StreamingTextWindowDataset(path, 4, tokenizer, rank=0, world_size=2))
    rank_one = list(StreamingTextWindowDataset(path, 4, tokenizer, rank=1, world_size=2))
    assert rank_zero
    assert rank_one
