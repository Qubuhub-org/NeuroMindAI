import torch

from neuromind import ModelConfig, NeuroMindLM


def tiny_config():
    return ModelConfig(
        vocab_size=256,
        context_length=16,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        dropout=0.0,
    )


def test_forward_shape_and_loss():
    model = NeuroMindLM(tiny_config())
    inputs = torch.randint(0, 256, (2, 8))
    logits, loss = model(inputs, inputs)
    assert logits.shape == (2, 8, 256)
    assert loss is not None and torch.isfinite(loss)


def test_context_limit_is_enforced():
    model = NeuroMindLM(tiny_config())
    inputs = torch.randint(0, 256, (1, 17))
    try:
        model(inputs)
    except ValueError as exc:
        assert "context window" in str(exc)
    else:
        raise AssertionError("expected a context length error")


def test_causal_mask_blocks_future_tokens():
    torch.manual_seed(7)
    model = NeuroMindLM(tiny_config()).eval()
    first = torch.tensor([[10, 11, 12, 13]])
    changed_future = torch.tensor([[10, 11, 99, 98]])
    with torch.no_grad():
        logits_a, _ = model(first)
        logits_b, _ = model(changed_future)
    assert torch.allclose(logits_a[:, :2], logits_b[:, :2], atol=1e-6)
