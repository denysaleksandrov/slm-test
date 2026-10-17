"""
Model 2: MLP language model (Bengio et al., 2003, "A Neural Probabilistic Language Model").

Idea: look at the last `context` characters (e.g. 8), not just one.
    1. turn each of the 8 characters into a learned vector (embedding), e.g. 64 numbers
    2. glue the 8 vectors together into one long vector (8 * 64 = 512 numbers)
    3. hidden layer + non-linearity, then an output layer with one score per character

New compared to bigram:
    - embeddings: characters that behave alike (vowels, punctuation) end up with
      similar vectors, so what the model learns about one transfers to the others
    - real context: "Juli" -> "e" is now learnable
What it cannot do:
    - see further back than `context` characters
    - share knowledge between positions: "a" as the 1st of 8 chars and "a" as the 5th
      go through completely different weights. Attention (model 3) fixes both.
Compared to MNIST: this is a 784 -> 10 -> 10 network, with the 784 pixels replaced
by 8 concatenated character embeddings.

Measured (mac preset, 0.3M params, ~2 min on an Apple M5): best validation loss 1.67 at step
~2000. After that the train loss keeps falling (1.21) while validation loss creeps back up
(1.70) - overfitting: it starts memorising the training text.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPLM(nn.Module):
    def __init__(self, vocab_size: int, context: int = 8, n_embd: int = 64, n_hidden: int = 512):
        super().__init__()
        self.context = context
        self.block_size = context  # generation only needs the last `context` tokens
        self.token_embedding = nn.Embedding(vocab_size, n_embd)  # id -> vector of n_embd
        self.hidden = nn.Linear(context * n_embd, n_hidden)       # concatenated window -> hidden
        self.output = nn.Linear(n_hidden, vocab_size)             # hidden -> next-char scores
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        emb = self.token_embedding(idx)  # (B, T, C)  C = n_embd

        # Build, for EVERY position t, the window of the `context` tokens ending at t.
        # Positions near the start don't have enough history: pad with zero vectors.
        emb = F.pad(emb, (0, 0, self.context - 1, 0))      # (B, context-1 + T, C) - pad on the left in time
        windows = emb.unfold(1, self.context, 1)           # (B, T, C, context) - sliding windows along time
        windows = windows.transpose(2, 3)                  # (B, T, context, C)
        x = windows.reshape(B, T, self.context * emb.size(-1))  # (B, T, context*C) - glue together

        x = torch.tanh(self.hidden(x))  # (B, T, n_hidden) - non-linearity (Bengio used tanh)
        logits = self.output(x)         # (B, T, vocab)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss
