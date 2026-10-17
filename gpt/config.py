"""
All hyperparameters in one place.

How the final config is built (later steps override earlier ones):
    1. defaults in the Config dataclass below
    2. preset         (--preset tiny | mac | gpt2-small)  - "how big / for what hardware"
    3. model defaults (--model bigram | mlp | attn | gpt) - e.g. bigram likes a higher LR
    4. command line   (any field, e.g. --n_layer 4 --learning_rate 3e-4)
"""

from dataclasses import dataclass, fields, asdict


@dataclass
class Config:
    # ---- what to train ----
    model: str = "gpt"                   # bigram | mlp | attn | gpt
    dataset: str = "shakespeare_char"    # folder name under data/
    out_dir: str = ""                    # checkpoints/logs; "" -> out/<dataset>-<model>-<preset>

    # ---- batches ----
    batch_size: int = 64                 # windows per micro-batch (per GPU)
    block_size: int = 256                # context length: how many tokens the model sees
    grad_accum_steps: int = 1            # micro-batches summed into one optimizer step

    # ---- optimisation ----
    max_iters: int = 5000                # optimizer steps
    learning_rate: float = 1e-3          # peak learning rate
    min_lr: float = 1e-4                 # LR at the end of the cosine decay
    warmup_iters: int = 100              # linear LR ramp-up from 0 at the start
    weight_decay: float = 0.1            # AdamW regularisation (shrinks weights a little)
    beta1: float = 0.9                   # AdamW momentum
    beta2: float = 0.99                  # AdamW 2nd-moment decay (0.95 for big runs)
    grad_clip: float = 1.0               # cap the gradient norm; 0 = off

    # ---- model size ----
    n_layer: int = 6                     # transformer blocks stacked (gpt)
    n_head: int = 6                      # attention heads per block (attn, gpt)
    n_embd: int = 384                    # embedding / hidden width (all models)
    dropout: float = 0.2                 # randomly zero activations while training
    mlp_context: int = 8                 # mlp model: how many previous tokens it sees
    mlp_hidden: int = 512                # mlp model: hidden layer width

    # ---- evaluation / logging ----
    eval_interval: int = 250             # measure train/val loss every N steps
    eval_iters: int = 100                # batches averaged per loss measurement
    log_interval: int = 50               # print training loss every N steps
    sample_tokens: int = 200             # length of the text sample printed at each eval

    # ---- system ----
    device: str = "auto"                 # auto | cuda | mps | cpu
    dtype: str = "auto"                  # auto | float32 | bfloat16 | float16
    compile: bool = False                # torch.compile the model (CUDA: big speed-up)
    fast_attn: bool = False              # fused attention kernel instead of the explicit one
    seed: int = 1337


PRESETS = {
    # A few seconds on CPU: is everything wired up correctly?
    "tiny": dict(
        batch_size=16, block_size=64, n_layer=2, n_head=4, n_embd=64, dropout=0.0,
        max_iters=300, eval_interval=100, eval_iters=20, log_interval=25,
        warmup_iters=20, sample_tokens=100,
    ),
    # The real Shakespeare run on a laptop / Apple Silicon: ~10.7M parameters for gpt.
    # Reference (Apple M5, 16GB): ~0.65 s/step -> ~1 hour for 5000 steps.
    # Memory (on Apple Silicon the GPU shares system RAM), measured on that M5 for gpt:
    #   batch 64 x accum 1: 5.2 GB | batch 32 x 2: 3.2 GB | batch 16 x 4: 2.2 GB
    # Same 64 sequences per optimizer step either way (identical math), ~2% slower.
    # eval_iters x4 so every loss estimate still averages 6,400 windows.
    # fast_attn: same results as explicit attention; on MPS it saves no memory or time
    # (PyTorch has no memory-efficient attention kernel for MPS), on CUDA it does both.
    "mac": dict(batch_size=16, grad_accum_steps=4, eval_iters=400, eval_interval=500, fast_attn=True),
    # GPT-2 small shape (~124M params) for TinyStories / FineWeb on CUDA GPUs with enough
    # memory (H100/H200/A100, RTX 6000-class). Hardware-neutral: on a 48GB card use
    # --batch_size 16 --grad_accum_steps 4 if 32 x 2 doesn't fit (same result).
    # Tokens per optimizer step = batch_size * block_size * grad_accum_steps * num_gpus
    #                           = 32 * 1024 * 2 * 8 GPUs = ~0.5M (the GPT-2 paper value)
    "gpt2-small": dict(
        dataset="tinystories", batch_size=32, block_size=1024, grad_accum_steps=2,
        n_layer=12, n_head=12, n_embd=768, dropout=0.0,
        max_iters=20000, learning_rate=6e-4, min_lr=6e-5, warmup_iters=700,
        weight_decay=0.1, beta2=0.95, eval_interval=500, eval_iters=100, log_interval=10,
        compile=True, fast_attn=True,
    ),
}

# Small models learn best with different settings than big ones.
MODEL_DEFAULTS = {
    "bigram": dict(learning_rate=1e-2, min_lr=1e-3, weight_decay=0.0, max_iters=3000),
    "mlp": dict(learning_rate=3e-3, min_lr=3e-4, weight_decay=0.0, n_embd=64),
    "attn": dict(),
    "gpt": dict(),
}


def build_config(preset: str, model: str, overrides: dict) -> Config:
    """defaults -> preset -> model defaults -> command-line overrides."""
    values = {}
    values.update(PRESETS[preset])
    values.update(MODEL_DEFAULTS[model])
    # for the tiny preset keep the short run even for bigram/mlp
    if preset == "tiny":
        values["max_iters"] = PRESETS["tiny"]["max_iters"]
    values.update({k: v for k, v in overrides.items() if v is not None})
    values["model"] = model
    cfg = Config(**values)
    if not cfg.out_dir:
        cfg.out_dir = f"out/{cfg.dataset}-{cfg.model}-{preset}"
    return cfg


def config_fields():
    """(name, type, default) for every field - used to generate command-line flags."""
    return [(f.name, f.type, f.default) for f in fields(Config)]


def to_dict(cfg: Config) -> dict:
    return asdict(cfg)
