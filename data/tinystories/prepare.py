"""
Prepare TinyStories (Hugging Face: roneneldan/TinyStories) with GPT-2 BPE tokens.

TinyStories: ~2.1M short children's stories written by GPT-3.5/4 using a small
vocabulary. Designed so that SMALL models (1M-30M parameters) can learn to write
coherent English - a great next step after Shakespeare, and big enough (~470M tokens)
to keep an H100 busy.

What this does:
  1. stream the stories from Hugging Face (no need to download everything first)
  2. tokenize with GPT-2's BPE (tiktoken): ~4 characters per token on average
  3. put an <|endoftext|> token after every story, so the model learns where a
     story ends and never mixes two stories up when a window spans both
  4. append everything to train.bin / val.bin (uint16 - GPT-2's 50,257 ids fit in 16 bits)

Usage:
    python data/tinystories/prepare.py                     # full dataset (~1GB download, a few minutes)
    python data/tinystories/prepare.py --max_stories 5000  # small subset to test the pipeline

Then train, e.g. on a laptop:
    python train.py --model gpt --preset mac --dataset tinystories
"""

import argparse
import os
import pickle
import time

import numpy as np
import tiktoken
from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
HF_DATASET = "roneneldan/TinyStories"
BATCH = 1024  # stories tokenized together (tiktoken uses several threads per batch)


def write_split(hf_split: str, out_name: str, enc, limit: int | None) -> int:
    """Tokenize one split and write it to <out_name>.bin. Returns the number of tokens."""
    # streaming=True: iterate over the data while it downloads, nothing kept in memory
    stories = load_dataset(HF_DATASET, split=hf_split, streaming=True)
    if limit:
        stories = stories.take(limit)

    path = os.path.join(HERE, f"{out_name}.bin")
    n_tokens, n_stories, t0 = 0, 0, time.time()
    with open(path, "wb") as f:
        batch = []
        for example in stories:
            batch.append(example["text"])
            if len(batch) == BATCH:
                n_tokens += flush(batch, f, enc)
                n_stories += len(batch)
                batch = []
                if n_stories % (BATCH * 100) == 0:
                    print(f"  {out_name}: {n_stories:,} stories, {n_tokens:,} tokens, {time.time() - t0:.0f}s")
        if batch:
            n_tokens += flush(batch, f, enc)
            n_stories += len(batch)
    print(f"{out_name}.bin: {n_stories:,} stories -> {n_tokens:,} tokens")
    return n_tokens


def flush(texts: list[str], f, enc) -> int:
    """Tokenize a batch of stories, add <|endoftext|> after each, append to the file."""
    token_lists = enc.encode_ordinary_batch(texts, num_threads=8)
    ids = []
    for tokens in token_lists:
        ids.extend(tokens)
        ids.append(enc.eot_token)  # story boundary
    np.array(ids, dtype=np.uint16).tofile(f)
    return len(ids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max_stories", type=int, default=None, help="only the first N train stories (testing)")
    args = p.parse_args()

    enc = tiktoken.get_encoding("gpt2")
    val_limit = max(args.max_stories // 20, 100) if args.max_stories else None
    write_split("train", "train", enc, args.max_stories)
    write_split("validation", "val", enc, val_limit)

    # sanity check: the first story decodes back to readable text
    first = np.memmap(os.path.join(HERE, "train.bin"), dtype=np.uint16, mode="r")[:60]
    print("first tokens decoded:", repr(enc.decode(first.tolist())))

    meta = {"tokenizer": "gpt2", "vocab_size": enc.n_vocab}
    with open(os.path.join(HERE, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    print(f"wrote train.bin, val.bin, meta.pkl to {HERE}")


if __name__ == "__main__":
    main()
