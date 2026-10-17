"""
Train any of the four models. One loop for everything, from a bigram on a laptop to
GPT-2-sized runs on 8 GPUs.

Examples
    # Laptop / Apple Silicon, Shakespeare:
    python train.py --model bigram --preset mac
    python train.py --model mlp    --preset mac
    python train.py --model attn   --preset mac
    python train.py --model gpt    --preset mac
    python train.py --model gpt    --preset tiny --device cpu       # quick wiring check
    python train.py --model gpt    --preset mac  --resume           # continue a run

    # 8 GPUs on one machine (DDP):
    torchrun --standalone --nproc_per_node=8 train.py --model gpt --preset h100

Any config field can be overridden: --n_layer 4 --learning_rate 3e-4 --max_iters 2000 ...

One training step (the same idea as any PyTorch MNIST training loop):
    x, y = get_batch()           # (B, T) inputs and next-token targets
    logits, loss = model(x, y)   # forward pass
    loss.backward()              # backward pass: gradients
    optimizer.step()             # update weights
    optimizer.zero_grad()
Everything else below is making that faster, more stable, or observable.
"""

import argparse
import csv
import math
import os
import time

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from gpt import build_model
from gpt.config import PRESETS, MODEL_DEFAULTS, build_config, config_fields, to_dict
from gpt.data import TokenData
from gpt.device import (
    autocast_context, cleanup_ddp, enable_fast_matmul, get_dtype, set_seed, setup_ddp,
)
from gpt.sampling import generate
from gpt.tokenizer import load_tokenizer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="gpt", choices=list(MODEL_DEFAULTS))
    p.add_argument("--preset", default="mac", choices=list(PRESETS))
    p.add_argument("--resume", action="store_true", help="continue from <out_dir>/ckpt.pt")
    # one flag per Config field, default None = "not given, keep preset value"
    for name, typ, _ in config_fields():
        if name == "model":
            continue
        if typ is bool or typ == "bool":
            p.add_argument(f"--{name}", action=argparse.BooleanOptionalAction, default=None)
        else:
            p.add_argument(f"--{name}", type={"int": int, "float": float, "str": str}.get(typ, typ), default=None)
    args = p.parse_args()
    overrides = {k: v for k, v in vars(args).items() if k not in ("model", "preset", "resume")}
    return args, build_config(args.preset, args.model, overrides)


def get_lr(it: int, cfg) -> float:
    """
    Learning-rate schedule:
      warmup: ramp up linearly from ~0 to learning_rate over warmup_iters steps.
              At the start weights are random and gradients large/noisy - big steps
              then can wreck the model.
      cosine decay: glide down to min_lr along half a cosine wave. Smaller steps at
              the end let the model settle into a good minimum instead of bouncing around.
    """
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / cfg.warmup_iters
    if it >= cfg.max_iters:
        return cfg.min_lr
    progress = (it - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)  # 0 -> 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))                             # 1 -> 0
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


def make_optimizer(model, cfg, device):
    """
    AdamW: Adam (per-weight adaptive step sizes + momentum) with weight decay done right.
    Weight decay (pulling weights slightly towards 0 each step) is applied only to
    matrices (2-D weights), not to biases or LayerNorm scales - shrinking those just hurts.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [
        {"params": [p for p in params if p.dim() >= 2], "weight_decay": cfg.weight_decay},
        {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
    ]
    # fused=True: one GPU kernel for the whole update (CUDA only) - faster
    extra = {"fused": True} if device.startswith("cuda") else {}
    return torch.optim.AdamW(groups, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), **extra)


@torch.no_grad()
def estimate_loss(model, data, cfg, device, ctx):
    """
    Average loss over eval_iters random batches, for train and val data.
    A single batch is noisy; the average gives a reliable number.
    train << val means the model is memorising (overfitting) instead of generalising.
    """
    out = {}
    model.eval()  # dropout off
    for split in ("train", "val"):
        losses = torch.zeros(cfg.eval_iters)
        for i in range(cfg.eval_iters):
            x, y = data.get_batch(split, cfg.batch_size, cfg.block_size, device)
            with ctx:
                _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def sample_text(model, tokenizer, device, n_tokens: int) -> str:
    """Generate a short sample from a newline (char) / end-of-text (BPE) token."""
    start_id = tokenizer.encode("\n")[0] if hasattr(tokenizer, "stoi") else tokenizer.eot
    idx = torch.tensor([[start_id]], dtype=torch.long, device=device)
    return tokenizer.decode(generate(model, idx, n_tokens, temperature=0.8, top_k=50)[0].tolist())


def main():
    args, cfg = parse_args()

    # ---------------- setup: devices, seeds, data ----------------
    is_ddp, rank, local_rank, world_size, device = setup_ddp(cfg.device)
    master = rank == 0                       # only rank 0 prints, evaluates and saves
    set_seed(cfg.seed + rank)                # different random batches on every GPU
    enable_fast_matmul()
    dtype = get_dtype(device, cfg.dtype)
    ctx = autocast_context(device, dtype)    # mixed precision (no-op for float32)

    data_dir = os.path.join("data", cfg.dataset)
    data = TokenData(data_dir)
    tokenizer = load_tokenizer(data_dir)
    # Pad the vocab to a multiple of 64 for big BPE vocabs (50257 -> 50304): GPU matrix
    # kernels are faster on "round" sizes. The extra ids are never used in the data.
    vocab_size = data.vocab_size if data.vocab_size < 1000 else math.ceil(data.vocab_size / 64) * 64

    tokens_per_step = cfg.batch_size * cfg.block_size * cfg.grad_accum_steps * world_size
    if master:
        os.makedirs(cfg.out_dir, exist_ok=True)
        print(f"device={device} dtype={str(dtype).replace('torch.', '')} world_size={world_size}")
        print(f"dataset={cfg.dataset}: {data.num_tokens('train'):,} train / {data.num_tokens('val'):,} val tokens, vocab {vocab_size}")
        print(f"tokens per optimizer step: {tokens_per_step:,}")

    # ---------------- model + optimizer ----------------
    model = build_model(cfg, vocab_size).to(device)
    optimizer = make_optimizer(model, cfg, device)
    start_iter, best_val = 0, float("inf")

    ckpt_path = os.path.join(cfg.out_dir, "ckpt.pt")
    if args.resume:
        # restore weights AND optimizer state (Adam's running averages) AND step counter,
        # so the run continues exactly where it stopped
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter, best_val = ckpt["iter"] + 1, ckpt["best_val"]
        if master:
            print(f"resumed from {ckpt_path} at step {ckpt['iter']}")

    raw_model = model  # keep a handle to the plain model for saving/sampling
    if master:
        print(f"model={cfg.model}: {sum(p.numel() for p in raw_model.parameters()) / 1e6:.2f}M parameters")

    if cfg.compile:
        # torch.compile traces the model and fuses many small GPU operations into few
        # big ones. Takes ~1 min to start, then typically 1.5-2x faster on CUDA.
        if master:
            print("compiling the model...")
        model = torch.compile(model)
    if is_ddp:
        # DDP hooks into backward(): gradients are averaged across GPUs automatically
        model = DDP(model, device_ids=[local_rank] if device.startswith("cuda") else None)

    # float16 needs loss scaling to avoid gradients underflowing to 0 (bf16/fp32 don't)
    scaler = torch.amp.GradScaler(enabled=(dtype == torch.float16))

    log_path = os.path.join(cfg.out_dir, "log.csv")
    if master and not args.resume:
        with open(log_path, "w", newline="") as f:
            csv.writer(f, lineterminator="\n").writerow(["iter", "train_loss", "val_loss", "lr", "elapsed_s"])

    # ---------------- training loop ----------------
    x, y = data.get_batch("train", cfg.batch_size, cfg.block_size, device)  # first batch
    t_start = t_log = time.time()
    for it in range(start_iter, cfg.max_iters + 1):
        lr = get_lr(it, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # --- evaluate, print a sample, save a checkpoint ---
        if master and (it % cfg.eval_interval == 0 or it == cfg.max_iters):
            losses = estimate_loss(raw_model, data, cfg, device, ctx)
            elapsed = time.time() - t_start
            print(f"step {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | lr {lr:.2e} | {elapsed / 60:.1f} min")
            with open(log_path, "a", newline="") as f:
                csv.writer(f, lineterminator="\n").writerow([it, f"{losses['train']:.4f}", f"{losses['val']:.4f}", f"{lr:.3e}", f"{elapsed:.1f}"])
            print("-" * 60 + "\n" + sample_text(raw_model, tokenizer, device, cfg.sample_tokens).strip() + "\n" + "-" * 60)
            improved = losses["val"] < best_val
            best_val = min(best_val, losses["val"])
            ckpt = {"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(), "iter": it,
                    "val_loss": losses["val"], "best_val": best_val, "config": to_dict(cfg),
                    "vocab_size": vocab_size}
            # ckpt.pt = latest state, for --resume. best.pt = lowest validation loss so far:
            # once a model starts overfitting, later checkpoints get WORSE on unseen text
            torch.save(ckpt, ckpt_path)
            if improved:
                torch.save(ckpt, os.path.join(cfg.out_dir, "best.pt"))
        if it == cfg.max_iters:
            break

        # --- one optimizer step = grad_accum_steps micro-batches ---
        # Gradient accumulation: run several small batches and ADD their gradients before
        # stepping. Same result as one big batch, without needing the memory for it.
        for micro in range(cfg.grad_accum_steps):
            if is_ddp:
                # only all-reduce gradients across GPUs on the last micro-batch (saves traffic)
                model.require_backward_grad_sync = micro == cfg.grad_accum_steps - 1
            with ctx:
                _, loss = model(x, y)
                loss = loss / cfg.grad_accum_steps  # average, not sum, over micro-batches
            # fetch the next batch now: on CUDA the copy overlaps with the backward pass
            x, y = data.get_batch("train", cfg.batch_size, cfg.block_size, device)
            scaler.scale(loss).backward()

        if cfg.grad_clip > 0:
            # if the total gradient is unusually large (a "spike"), scale it down to
            # grad_clip so a single bad batch can't throw the weights far off
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # --- progress line ---
        if master and it % cfg.log_interval == 0 and it > start_iter:
            now = time.time()
            dt = (now - t_log) / cfg.log_interval
            t_log = now
            # loss.item() waits for the GPU to finish, so the timing is honest
            print(f"  iter {it:5d} | loss {loss.item() * cfg.grad_accum_steps:.4f} | "
                  f"{dt * 1000:.0f} ms/step | {tokens_per_step / dt:,.0f} tok/s")

    if master:
        print(f"done: best val loss {best_val:.4f} (best.pt), {(time.time() - t_start) / 60:.1f} min, latest: {ckpt_path}")
    cleanup_ddp(is_ddp)


if __name__ == "__main__":
    main()
