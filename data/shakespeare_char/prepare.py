"""
Prepare the TinyShakespeare dataset for character-level language modelling.

What this does (run once):
  1. download ~1MB of Shakespeare plays as one plain-text file (input.txt)
  2. build a "vocabulary": the sorted list of every distinct character in the text
     -> each character gets an integer id (its position in that list)
  3. turn the whole text into a long list of ids ("tokens") and split it 90% / 10%
     into training data and validation data (text the model never trains on)
  4. save the ids as flat binary files train.bin / val.bin (uint16, 2 bytes per token)
     plus meta.pkl with the vocabulary so we can turn ids back into text later

Why binary files: training reads random windows straight from disk with np.memmap,
so the same code works for 1MB of Shakespeare and for 20GB of web text on the H100s.

Usage:
    python data/shakespeare_char/prepare.py
"""

import os
import pickle
import urllib.request

import numpy as np

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
HERE = os.path.dirname(os.path.abspath(__file__))  # write everything next to this script


def main():
    # 1. Download the text (only once - skip if it is already there)
    input_path = os.path.join(HERE, "input.txt")
    if not os.path.exists(input_path):
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, input_path)
    with open(input_path, encoding="utf-8") as f:
        text = f.read()
    print(f"length of dataset in characters: {len(text):,}")

    # 2. Vocabulary: every distinct character, sorted so the ids are stable between runs
    chars = sorted(set(text))
    vocab_size = len(chars)
    print(f"vocabulary ({vocab_size} chars): {''.join(chars)!r}")

    # Two lookup tables: character -> id ("string to int") and id -> character
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # 3. Encode the full text as ids and split: first 90% train, last 10% validation.
    #    A contiguous split (not random lines) means validation is genuinely "unseen" text.
    ids = np.array([stoi[c] for c in text], dtype=np.uint16)
    n = int(0.9 * len(ids))
    train_ids, val_ids = ids[:n], ids[n:]
    print(f"train has {len(train_ids):,} tokens, val has {len(val_ids):,} tokens")

    # Sanity check: decoding the ids must give back exactly the original text
    assert "".join(itos[int(i)] for i in ids[:1000]) == text[:1000], "encode/decode round-trip failed"

    # 4. Save: raw uint16 arrays (read later with np.memmap) + the vocabulary
    train_ids.tofile(os.path.join(HERE, "train.bin"))
    val_ids.tofile(os.path.join(HERE, "val.bin"))
    meta = {"tokenizer": "char", "vocab_size": vocab_size, "itos": itos, "stoi": stoi}
    with open(os.path.join(HERE, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    print(f"wrote train.bin, val.bin, meta.pkl to {HERE}")


if __name__ == "__main__":
    main()
