"""
Hardware helpers: pick the GPU, choose number precision, set up multi-GPU (DDP).

The same code runs on:
  - Apple Silicon Macs  -> "mps"  (Apple GPU via Metal Performance Shaders)
  - NVIDIA GPUs         -> "cuda" (e.g. H100/H200/A100)
  - anything else       -> "cpu"
"""

import os
import random
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist


def get_device(requested: str = "auto") -> str:
    """Best available device: cuda > mps > cpu (or exactly what was requested)."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype(device: str, requested: str = "auto") -> torch.dtype:
    """
    Precision used for the forward pass (via autocast).

    float32  - 4 bytes/number, the safe default. Used on Mac (MPS) and CPU.
    bfloat16 - 2 bytes/number, same range as float32 but less precision. H100s have
               dedicated hardware for it: ~2x faster and half the memory, and training
               quality is practically the same. Weights themselves stay float32.
    float16  - 2 bytes but a small range; needs loss scaling (GradScaler). Only for
               older NVIDIA GPUs without bfloat16.
    """
    names = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    if requested != "auto":
        return names[requested]
    if device.startswith("cuda") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def autocast_context(device: str, dtype: torch.dtype):
    """
    Context manager that runs matrix multiplies in `dtype` (mixed precision).
    For float32 there is nothing to do, so we return a no-op context.
    """
    if dtype == torch.float32:
        return nullcontext()
    device_type = "cuda" if device.startswith("cuda") else device
    return torch.autocast(device_type=device_type, dtype=dtype)


def set_seed(seed: int):
    """Make runs repeatable: same seed -> same initial weights and same batches."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds cuda and mps


def enable_fast_matmul():
    """
    On NVIDIA GPUs allow TF32 for float32 matrix multiplies (Ampere/Hopper tensor cores).
    Small precision loss, large speed-up. Harmless no-op elsewhere.
    """
    torch.set_float32_matmul_precision("high")


# --------------------------------------------------------------------------------------
# Multi-GPU: DistributedDataParallel (DDP)
#
# How DDP works: `torchrun --nproc_per_node=8 train.py` starts 8 copies of the script,
# one per GPU. Each copy ("rank") holds a full copy of the model, trains on different
# random batches, and after every backward pass the gradients are averaged across all
# GPUs (all-reduce), so every copy takes the same optimizer step and the weights stay
# identical. Result: ~8x more data per step.
#
# torchrun tells each process who it is via environment variables:
#   RANK        - global id of this process (0..WORLD_SIZE-1); rank 0 is the "master"
#   LOCAL_RANK  - id on this machine -> which GPU to use
#   WORLD_SIZE  - total number of processes
# --------------------------------------------------------------------------------------

def setup_ddp(requested_device: str = "auto"):
    """
    Returns (is_ddp, rank, local_rank, world_size, device).
    Without torchrun this is a normal single-process run on the best device.
    """
    if "RANK" not in os.environ:  # plain `python train.py`
        return False, 0, 0, 1, get_device(requested_device)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if torch.cuda.is_available():
        # NVIDIA: NCCL is the fast GPU-to-GPU communication library (NVLink / InfiniBand)
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl")
    else:
        # Mac / CPU: "gloo" backend on CPU. MPS does not support DDP, so this mode is
        # only for testing the multi-GPU code path locally before using the H100s.
        device = "cpu"
        dist.init_process_group(backend="gloo")
    return True, rank, local_rank, world_size, device


def cleanup_ddp(is_ddp: bool):
    if is_ddp:
        dist.destroy_process_group()
