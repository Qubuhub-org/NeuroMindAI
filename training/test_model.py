import torch
from neuromind import NeuroMindConfig, NeuroMindForCausalLM

def test_forward_shapes():
    config = NeuroMindConfig(vocab_size=128, hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_kv_heads=2, intermediate_size=256, max_position_embeddings=32)
    model = NeuroMindForCausalLM(config)
    out = model(torch.randint(0, config.vocab_size, (2, 8)))
    assert out["logits"].shape == (2, 8, config.vocab_size)

def test_backward():
    config = NeuroMindConfig(vocab_size=64, hidden_size=32, num_hidden_layers=1, num_attention_heads=4, num_kv_heads=2, intermediate_size=128, max_position_embeddings=16)
    model = NeuroMindForCausalLM(config)
    ids = torch.randint(0, config.vocab_size, (2, 8))
    labels = torch.randint(0, config.vocab_size, (2, 8))
    loss = model(ids, labels=labels)["loss"]
    loss.backward()
    assert torch.isfinite(loss)



def test_rejects_invalid_rope_head_dimension():
    import pytest
    with pytest.raises(ValueError, match="even for RoPE"):
        NeuroMindConfig(vocab_size=32, hidden_size=30, num_hidden_layers=1,
                        num_attention_heads=2, num_kv_heads=1, intermediate_size=64)


def test_rejects_context_overflow():
    import pytest
    config = NeuroMindConfig(vocab_size=32, hidden_size=32, num_hidden_layers=1,
                             num_attention_heads=2, num_kv_heads=1,
                             intermediate_size=64, max_position_embeddings=4)
    model = NeuroMindForCausalLM(config)
    with pytest.raises(ValueError, match="exceeds max_position_embeddings"):
        model(torch.randint(0, config.vocab_size, (1, 5)))


def test_generation_handles_long_prompt_with_context_window():
    config = NeuroMindConfig(vocab_size=32, hidden_size=32, num_hidden_layers=1,
                             num_attention_heads=2, num_kv_heads=1,
                             intermediate_size=64, max_position_embeddings=4,
                             eos_token_id=None)
    model = NeuroMindForCausalLM(config)
    prompt = torch.randint(0, config.vocab_size, (1, 6))
    generated = model.generate(prompt, max_new_tokens=2, temperature=0)
    assert generated.shape == (1, 8)
