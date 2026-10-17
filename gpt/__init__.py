"""mini-GPT: from a bigram table to a transformer. See README.md."""

from .bigram import BigramLM
from .gpt import GPT, AttentionLM
from .mlp_lm import MLPLM


def build_model(cfg, vocab_size: int):
    """Create the model named by cfg.model with the sizes from the config."""
    if cfg.model == "bigram":
        return BigramLM(vocab_size)
    if cfg.model == "mlp":
        return MLPLM(vocab_size, context=cfg.mlp_context, n_embd=cfg.n_embd, n_hidden=cfg.mlp_hidden)
    if cfg.model == "attn":
        return AttentionLM(vocab_size, cfg.block_size, cfg.n_embd, cfg.n_head, cfg.dropout, cfg.fast_attn)
    if cfg.model == "gpt":
        return GPT(vocab_size, cfg.block_size, cfg.n_embd, cfg.n_head, cfg.n_layer, cfg.dropout, cfg.fast_attn)
    raise ValueError(f"unknown model {cfg.model!r} (bigram | mlp | attn | gpt)")
