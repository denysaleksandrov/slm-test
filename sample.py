"""
Generate text from a trained checkpoint.

    python sample.py --ckpt_dir out/shakespeare_char-gpt-mac
    python sample.py --ckpt_dir out/shakespeare_char-gpt-mac --prompt "ROMEO:" --num_samples 3
    python sample.py --ckpt_dir out/shakespeare_char-gpt-mac --temperature 0.5 --top_k 10

temperature < 1 = safer/more repetitive, > 1 = wilder; top_k = only the k likeliest tokens.
"""

import argparse
import os
import types

import torch

from gpt import build_model
from gpt.device import get_device, set_seed
from gpt.sampling import generate
from gpt.tokenizer import load_tokenizer


def load_model(ckpt_dir: str, device: str, which: str = "best"):
    """
    Rebuild the model from the config stored in the checkpoint and load its weights.
    which="best": lowest validation loss (best.pt); "latest": last saved step (ckpt.pt).
    Falls back to ckpt.pt for runs made before best.pt existed.
    """
    path = os.path.join(ckpt_dir, "best.pt" if which == "best" else "ckpt.pt")
    if not os.path.exists(path):
        path = os.path.join(ckpt_dir, "ckpt.pt")
    ckpt = torch.load(path, map_location=device)
    cfg = types.SimpleNamespace(**ckpt["config"])
    model = build_model(cfg, ckpt["vocab_size"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, cfg, ckpt


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt_dir", required=True, help="folder with ckpt.pt, e.g. out/shakespeare_char-gpt-mac")
    p.add_argument("--prompt", default="\n", help="text to continue")
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--seed", type=int, default=None, help="fix for repeatable output")
    p.add_argument("--which", default="best", choices=["best", "latest"], help="best.pt or ckpt.pt")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = get_device(args.device)
    if args.seed is not None:
        set_seed(args.seed)
    model, cfg, ckpt = load_model(args.ckpt_dir, device, args.which)
    tokenizer = load_tokenizer(os.path.join("data", cfg.dataset))
    if "val_loss" in ckpt:
        loss_info = f"val loss {ckpt['val_loss']:.4f}"
    else:  # checkpoints saved before val_loss was recorded
        loss_info = f"best val loss during the run {ckpt['best_val']:.4f}"
    print(f"# {cfg.model} at step {ckpt['iter']}, {loss_info}, on {device}")

    start = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    for i in range(args.num_samples):
        out = generate(model, start, args.max_new_tokens, args.temperature, args.top_k)
        print(f"\n===== sample {i + 1} =====")
        print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
