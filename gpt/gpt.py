"""
Models 3 and 4: self-attention and the GPT transformer.

----------------------------------------------------------------------------------
The problem with the previous models
    bigram: sees 1 token.  MLP: sees a fixed window, and treats "the same letter at a
    different position" with completely separate weights.

Self-attention (the core idea of transformers)
    Every token looks at all the tokens BEFORE it and decides itself which ones matter.
    Each token produces three vectors from its embedding:
        query  q: "what am I looking for?"
        key    k: "what do I contain?"
        value  v: "what will I hand over if you pick me?"
    Token t compares its query with the keys of tokens 0..t (dot product = similarity),
    turns those similarities into weights with softmax, and takes the weighted average
    of their values. E.g. in "Romeo and Juli|", the last "i" can put a big weight on
    "Romeo" and "Juli" and conclude "et" is next.

    causal mask: token t may NOT look at tokens after t - otherwise during training it
    could just read the answer. Implemented by setting those scores to -inf before
    softmax, which makes their weight exactly 0.

    multi-head: several attentions in parallel, each with its own q/k/v (e.g. one head
    tracks "who is speaking", another "are we inside a word"). Their outputs are
    concatenated.

GPT = a stack of transformer blocks, each block:
    x = x + attention(layernorm(x))     <- tokens exchange information
    x = x + feedforward(layernorm(x))   <- each token "thinks" about what it gathered
    residual "x + ...": each block only adds a correction; gradients flow straight
                        through the additions, which is what makes deep stacks trainable
    layernorm:          rescales each token vector to mean 0 / std 1, keeps numbers stable

Shapes used in comments:  B = batch, T = time (tokens), C = channels (n_embd),
                          nh = number of heads, hs = head size (C / nh), V = vocab size
----------------------------------------------------------------------------------
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Running nh separate single-head attentions and concatenating them is exactly what
    this does - but in one go: all heads' q, k, v come out of ONE linear layer and are
    then split into heads. Same math, much faster on a GPU.

    fast=False: explicit math (q @ k^T, mask, softmax, @ v) - readable, and lets us
                keep the attention weights for visualisation.
    fast=True:  F.scaled_dot_product_attention - fused "FlashAttention" kernel, same
                result, much less memory and much faster (needed on the H100s).
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float, fast: bool = False):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head, self.n_embd, self.fast = n_head, n_embd, fast
        self.dropout = dropout
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)  # q, k, v for all heads at once
        self.proj = nn.Linear(n_embd, n_embd, bias=False)     # mixes the heads' outputs
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        # Lower-triangular matrix of ones: row t has ones in columns 0..t = "allowed to look".
        # register_buffer: part of the module (moves with .to(device)) but not a weight.
        self.register_buffer("causal_mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
        self.store_attention = False  # set True to keep weights for plotting (explicit path)
        self.last_attention = None

    def forward(self, x):
        B, T, C = x.shape
        hs = C // self.n_head

        # 1. project every token to its query, key and value (all heads together)
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)          # 3 x (B, T, C)
        # 2. split C into nh heads of size hs and move heads next to batch
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)        # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)        # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)        # (B, nh, T, hs)

        if self.fast:
            # same as the explicit branch below, as one fused GPU kernel
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
            )                                                    # (B, nh, T, hs)
        else:
            # 3. similarity of every query with every key. Divided by sqrt(hs) so the
            #    scores don't grow with head size (big scores -> softmax becomes one-hot
            #    -> tiny gradients).
            att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)      # (B, nh, T, T)
            # 4. causal mask: no peeking at future tokens
            att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
            # 5. scores -> weights; each row sums to 1 over the allowed (past) tokens
            att = F.softmax(att, dim=-1)                         # (B, nh, T, T)
            if self.store_attention:
                self.last_attention = att.detach()
            att = self.attn_dropout(att)
            # 6. weighted average of the values
            y = att @ v                                          # (B, nh, T, hs)

        # 7. glue the heads back together and mix them
        y = y.transpose(1, 2).contiguous().view(B, T, C)         # (B, T, C)
        return self.resid_dropout(self.proj(y))


class FeedForward(nn.Module):
    """
    Per-token MLP: after attention gathered information from other tokens, each token
    processes it on its own. Expands to 4x width, GELU non-linearity, back to C.
    Most of a GPT's parameters live here.
    """

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),  # (B, T, 4C)
            nn.GELU(),                                   # smooth version of ReLU
            nn.Linear(4 * n_embd, n_embd, bias=False),  # (B, T, C)
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """One transformer block: communicate (attention), then compute (feed-forward)."""

    def __init__(self, n_embd, n_head, block_size, dropout, fast_attn):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout, fast_attn)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # (B, T, C) residual + attention
        x = x + self.ffwd(self.ln2(x))   # (B, T, C) residual + feed-forward
        return x


class _LMBase(nn.Module):
    """Shared parts of models 3 and 4: embeddings in, vocabulary scores out."""

    def __init__(self, vocab_size, block_size, n_embd, dropout):
        super().__init__()
        self.block_size = block_size
        # token embedding: WHAT the token is
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        # position embedding: WHERE it is (attention by itself has no notion of order -
        # without this, "dog bites man" and "man bites dog" would look the same)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.ln_f = nn.LayerNorm(n_embd)                       # final normalisation
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)  # vector -> next-token scores

    def _init_weights(self, module):
        # small random weights: every next token starts out equally likely (loss ~ ln V)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def embed(self, idx):
        B, T = idx.shape
        assert T <= self.block_size, f"sequence of {T} tokens > block_size {self.block_size}"
        pos = torch.arange(T, device=idx.device)                 # (T,) = 0, 1, ..., T-1
        x = self.token_embedding(idx) + self.position_embedding(pos)  # (B, T, C) + (T, C) broadcast
        return self.drop(x)

    def head(self, x, targets):
        x = self.ln_f(x)                                         # (B, T, C)
        logits = self.lm_head(x)                                 # (B, T, V)
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def set_store_attention(self, on: bool):
        """
        Keep the attention weights of the next forward pass (for plotting).
        The fused kernel never materialises them, so switch to the explicit path while
        storing, and restore the previous setting afterwards. Same results either way.
        """
        for m in self.modules():
            if isinstance(m, CausalSelfAttention):
                if on:
                    m._fast_before, m.fast = m.fast, False
                elif hasattr(m, "_fast_before"):
                    m.fast = m._fast_before
                m.store_attention = on


class AttentionLM(_LMBase):
    """
    Model 3: embeddings -> ONE multi-head self-attention layer -> scores.
    No feed-forward, no stacking. Use --n_head 1 to see a single head.

    New compared to MLP: any token can use any earlier token (up to block_size),
    with the same weights for every position.
    Measured (mac preset): validation loss 2.10 - WORSE than the MLP's 1.67. Attention
    only gathers information (a weighted average of earlier tokens); with no
    non-linear feed-forward layer nothing processes what was gathered.
    Missing: that per-token processing (feed-forward) and depth -> model 4.
    """

    def __init__(self, vocab_size, block_size, n_embd, n_head, dropout=0.0, fast_attn=False):
        super().__init__(vocab_size, block_size, n_embd, dropout)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout, fast_attn)
        self.apply(self._init_weights)

    def forward(self, idx, targets=None):
        x = self.embed(idx)          # (B, T, C)
        x = x + self.attn(x)         # (B, T, C)
        return self.head(x, targets)


class GPT(_LMBase):
    """
    Model 4: the full GPT. Same architecture family as GPT-2/3 - just smaller.
    Expected validation loss on Shakespeare (mac preset, ~10.7M params): ~1.5
    """

    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout=0.0, fast_attn=False):
        super().__init__(vocab_size, block_size, n_embd, dropout)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size, dropout, fast_attn) for _ in range(n_layer)]
        )
        # Weight tying: the input embedding (id -> vector) and the output layer
        # (vector -> score per id) share one matrix. Fewer parameters, and it works
        # better: tokens with similar meaning get similar vectors on both ends.
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
        # GPT-2 trick: shrink the init of layers that write into the residual stream,
        # so the sum of n_layer corrections doesn't blow up at the start of training
        for name, p in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("net.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def forward(self, idx, targets=None):
        x = self.embed(idx)          # (B, T, C)
        for block in self.blocks:
            x = block(x)             # (B, T, C)
        return self.head(x, targets)
