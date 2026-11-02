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


def test_native_neuromind_v1_uses_rope_and_swiglu():
    config = tiny_config()
    model = NeuroMindLM(config)
    assert model.config.architecture == "neuromind_v1"
    assert not hasattr(model, "position_embedding")
    inputs = torch.randint(0, config.vocab_size, (2, 8))
    logits, loss = model(inputs, inputs)
    assert logits.shape == (2, 8, config.vocab_size)
    assert loss is not None and torch.isfinite(loss)


def test_legacy_transformer_architecture_remains_loadable():
    config = tiny_config()
    config.architecture = "legacy_transformer"
    model = NeuroMindLM(config)
    inputs = torch.randint(0, config.vocab_size, (2, 8))
    logits, loss = model(inputs, inputs)
    assert logits.shape == (2, 8, config.vocab_size)
    assert loss is not None and torch.isfinite(loss)

def test_kv_cache_matches_full_sequence_logits():
    torch.manual_seed(11)
    model = NeuroMindLM(tiny_config()).eval()
    tokens = torch.randint(0, 256, (2, 9))
    with torch.no_grad():
        full_logits, _ = model(tokens)
        prefix_logits, cache = model.forward_with_cache(tokens[:, :8])
        next_logits, updated_cache = model.forward_with_cache(tokens[:, 8:], cache)
    assert torch.allclose(prefix_logits, full_logits[:, :8], atol=1e-5, rtol=1e-5)
    assert torch.allclose(next_logits[:, -1], full_logits[:, -1], atol=1e-5, rtol=1e-5)
    assert updated_cache[0][0].size(-2) == tokens.size(1)


def test_kv_cache_rejects_context_overflow():
    model = NeuroMindLM(tiny_config()).eval()
    tokens = torch.randint(0, 256, (1, 16))
    _, cache = model.forward_with_cache(tokens)
    try:
        model.forward_with_cache(torch.tensor([[1]]), cache)
    except ValueError as exc:
        assert "context window" in str(exc)
    else:
        raise AssertionError("expected cache context overflow")
