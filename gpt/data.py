"""
Feeding the model: random training windows from the token files.

A language model learns "given these tokens, what comes next?". From one window of
block_size + 1 consecutive tokens we get block_size training examples at once:

    text   : F  i  r  s  t     C  i      (block_size = 8, plus one extra token)
    x      : F  i  r  s  t     C         <- input
    y      :    i  r  s  t     C  i      <- target = input shifted left by one

    position 0: after "F"            predict "i"
    position 1: after "Fi"           predict "r"
    ...
    position 7: after "First Ci"     predict "i"

So y is just x shifted by one token. The model sees every prefix length from 1 to
block_size, which is what lets it generate text later starting from a short prompt.
"""

import os
import pickle

import numpy as np
import torch


class TokenData:
    """Random batches of (x, y) windows from data/<dataset>/{train,val}.bin."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
            self.meta = pickle.load(f)
        self.vocab_size = self.meta["vocab_size"]

    def num_tokens(self, split: str) -> int:
        return len(self._open(split))

    def _open(self, split: str) -> np.memmap:
        # np.memmap maps the file into memory without reading it: only the windows we
        # index are actually loaded. Re-opened on every call on purpose - keeping one
        # memmap alive for hours slowly leaks memory (known numpy behaviour).
        return np.memmap(os.path.join(self.data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")

    def get_batch(self, split: str, batch_size: int, block_size: int, device: str):
        """
        Returns x, y: both int64 tensors of shape (batch_size, block_size) on `device`.
        Each row is an independent window starting at a random position.
        """
        data = self._open(split)
        # random start positions; -block_size so that the +1 target still fits
        starts = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in starts])
        y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in starts])

        if device.startswith("cuda"):
            # pinned (page-locked) CPU memory allows an asynchronous copy to the GPU,
            # so the next batch is transferred while the GPU is still busy computing
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y
