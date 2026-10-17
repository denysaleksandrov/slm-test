"""
Tokenizers: convert text <-> list of integer ids.

Two kinds are used in this project:

CharTokenizer (TinyShakespeare)
    one character = one token, vocabulary of 65. Simple and easy to inspect,
    but the model has to spend effort just learning how to spell.

BPETokenizer (TinyStories / FineWeb, stage 5)
    GPT-2's byte-pair encoding: frequent chunks of text become single tokens,
    e.g. " the" or "ing". Vocabulary of 50,257. Text becomes ~4x fewer tokens, so the
    same context window covers ~4x more text and the model can focus on meaning.
    Uses OpenAI's `tiktoken` library - the exact tokenizer GPT-2 was trained with.
"""

import os
import pickle


class CharTokenizer:
    def __init__(self, stoi: dict, itos: dict):
        self.stoi, self.itos = stoi, itos
        self.vocab_size = len(itos)

    def encode(self, text: str) -> list[int]:
        unknown = sorted({c for c in text if c not in self.stoi})
        if unknown:
            raise ValueError(f"characters not in the training vocabulary: {unknown}")
        return [self.stoi[c] for c in text]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


class BPETokenizer:
    def __init__(self, name: str = "gpt2"):
        import tiktoken  # imported here so char-level runs don't need it

        self.enc = tiktoken.get_encoding(name)
        self.vocab_size = self.enc.n_vocab  # 50257 for gpt2
        self.eot = self.enc.eot_token  # <|endoftext|>, separates documents

    def encode(self, text: str) -> list[int]:
        # encode_ordinary: treat text literally, never produce special tokens from it
        return self.enc.encode_ordinary(text)

    def decode(self, ids) -> str:
        # ids >= n_vocab can appear if the model's vocab was padded (see train.py); drop them
        return self.enc.decode([int(i) for i in ids if int(i) < self.vocab_size])


def load_tokenizer(data_dir: str):
    """Build the right tokenizer from data/<dataset>/meta.pkl."""
    with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    if meta["tokenizer"] == "char":
        return CharTokenizer(meta["stoi"], meta["itos"])
    return BPETokenizer(meta["tokenizer"])
