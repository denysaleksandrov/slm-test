"""
Model 1: the Bigram language model - the simplest possible language model.

Idea: predict the next character using ONLY the current character.
The whole model is one table of size (vocab_size x vocab_size):
    row = current character, column = score for each possible next character.
After training, row "q" puts almost all its weight on "u", row "." on " " or "\n", etc.

What it learns:      which letters tend to follow which.
What it cannot do:   use any earlier context. After "Romeo and Juli" it only sees "i"
                     and has no idea that "et" is coming.
Compared to MNIST:   like a 784 -> 10 MNIST network: a single linear map from input to
                     class scores. Here the "classes" are the 65 possible next characters.

Expected: loss starts at ~4.17 (= ln 65, i.e. a uniform guess among 65 chars)
and ends around 2.5 on validation data.
"""

import torch.nn as nn
import torch.nn.functional as F


class BigramLM(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        # nn.Embedding is a lookup table: token id -> one row of learnable numbers.
        # Here each row has vocab_size numbers = the next-token scores (logits) directly.
        self.token_logits = nn.Embedding(vocab_size, vocab_size)
        # Start near zero so all next tokens are equally likely (loss = ln(vocab_size))
        nn.init.normal_(self.token_logits.weight, mean=0.0, std=0.02)
        self.block_size = 1  # it only ever looks at the last token

    def forward(self, idx, targets=None):
        # idx: (B, T) token ids   B = batch size, T = time (sequence length)
        logits = self.token_logits(idx)  # (B, T, vocab) - one score vector per position

        loss = None
        if targets is not None:
            # cross_entropy wants (N, classes) and (N,): flatten batch and time together.
            # It applies softmax and measures how much probability the correct
            # next token got: loss = -log(p_correct), averaged over all positions.
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss
