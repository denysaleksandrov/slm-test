"""
Generating text: the same loop works for every model in this project.

The model outputs, for each position, a score ("logit") for every token in the
vocabulary. To generate:
    1. feed the text so far (at most block_size tokens - the model's context window)
    2. take the scores for the LAST position = "what comes next?"
    3. turn scores into probabilities with softmax
    4. randomly pick a token according to those probabilities
    5. append it to the text, repeat

temperature: divides the scores before softmax.
    < 1.0 -> sharper distribution, safer/more repetitive text
    > 1.0 -> flatter distribution, more surprising/more nonsense text
    0     -> always pick the most likely token (greedy)
top_k: only sample from the k most likely tokens (cuts off the long tail of junk).
"""

import torch
import torch.nn.functional as F


@torch.no_grad()  # no gradients needed: we are not training here (saves memory + time)
def generate(model, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
    """
    idx: (B, T) tensor of token ids to continue from.
    Returns (B, T + max_new_tokens).
    """
    was_training = model.training
    model.eval()  # switch off dropout
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.block_size:]          # crop to the context window
        logits, _ = model(idx_cond)                    # (B, T, vocab)
        logits = logits[:, -1, :]                      # (B, vocab) - last position only

        if temperature <= 0:                           # greedy decoding
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                # value of the k-th largest score per row; everything below -> -inf
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits[logits < kth] = -float("inf")
            probs = F.softmax(logits, dim=-1)          # (B, vocab), rows sum to 1
            next_id = torch.multinomial(probs, num_samples=1)  # (B, 1) random draw

        idx = torch.cat([idx, next_id], dim=1)         # (B, T+1)
    model.train(was_training)
    return idx
