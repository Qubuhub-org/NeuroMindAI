import torch
from torch import nn
import torch.nn.functional as F
from .block import NeuroMindBlock
from .config import NeuroMindConfig
from .normalization import RMSNorm

class NeuroMindTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([NeuroMindBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class NeuroMindForCausalLM(nn.Module):
    def __init__(self, config: NeuroMindConfig):
        super().__init__()
        self.config = config
        self.transformer = NeuroMindTransformer(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.transformer.embed_tokens.weight

    def forward(self, input_ids, labels=None):
        hidden = self.transformer(input_ids)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=32, temperature=1.0, top_k=0):
        if input_ids.ndim != 2 or input_ids.size(1) == 0:
            raise ValueError("input_ids must have shape [batch, non-zero sequence]")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")

        self.eval()
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.max_position_embeddings:]
            logits = self(context)["logits"][:, -1, :]
            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < values[:, [-1]]] = -float("inf")
                next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if self.config.eos_token_id is not None and (next_token == self.config.eos_token_id).all():
                break
        return input_ids
