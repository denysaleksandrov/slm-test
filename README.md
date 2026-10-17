# mini-GPT: a text generator in PyTorch, from a bigram table to a transformer

Train a language model on text, and it learns to continue text one character
(or token) at a time. There are four models; each one adds one idea, and all four
share the same data, training loop and sampling code:

| `--model` | idea | params | val loss* | file |
|---|---|---|---|---|
| `bigram` | next char from the current char only (a 65×65 table) | 4K | 2.49 | `gpt/bigram.py` |
| `mlp` | embeddings + 8 chars of context + a hidden layer (Bengio 2003) | 0.3M | 1.67 | `gpt/mlp_lm.py` |
| `attn` | one multi-head self-attention layer, nothing else | 0.7M | 2.10 | `gpt/gpt.py` |
| `gpt` | 6 transformer blocks (attention + feed-forward, residuals, LayerNorm) | 10.7M | **1.46** | `gpt/gpt.py` |

\* Reference results on TinyShakespeare with `--preset mac`. Lower is better; `ln(65) = 4.17`
is a random guess. `notebooks/explore.ipynb` plots the runs side by side and explains the
surprises:

- **The MLP overfits.** Its validation loss turns back up after ~2000 steps.
- **Attention alone loses to the MLP.** Without a feed-forward layer, nothing processes what
  attention gathers.
- **The GPT peaks at step ~1500, then overfits.** By step 5000 it reaches train 0.58 / val 1.72,
  because 10.7M parameters is a lot for 1M characters of text. `best.pt` keeps the checkpoint
  with the lowest validation loss, and `--max_iters 2500` stops before the overfitting really
  starts (val 1.46 at the end of the run).

The code is written for learning: every step is commented, and the model code notes the tensor
shape on each line.

## Setup

Python 3.10 or newer (developed with 3.14).

```bash
git clone <this-repo-url> mini-gpt && cd mini-gpt
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data/shakespeare_char/prepare.py               # downloads ~1MB, writes train.bin/val.bin
```

The datasets and training outputs (`data/*/*.bin`, `out/`) are not in the repo. The prepare
scripts recreate the data, and training recreates the checkpoints.

## Train and generate

```bash
python train.py --model bigram --preset mac     # seconds
python train.py --model mlp    --preset mac     # a few minutes
python train.py --model attn   --preset mac     # ~5-10 min on a laptop GPU
python train.py --model gpt    --preset mac     # ~1-1.5 h on a laptop GPU (see below)

python sample.py --ckpt_dir out/shakespeare_char-gpt-mac --prompt "ROMEO:" --num_samples 2
jupyter notebook notebooks/explore.ipynb
```

- `--preset tiny --device cpu` checks in seconds that everything is wired up.
- `--resume` continues a stopped run from `out/<run>/ckpt.pt`.
- Every field in `gpt/config.py` is a flag, e.g. `--n_layer 4 --learning_rate 3e-4 --max_iters 2000`.
- The device is picked automatically: `cuda` > `mps` (Apple Silicon) > `cpu`. `--device` overrides it.
- Each run writes to `out/<dataset>-<model>-<preset>/` (or `--out_dir`):
  - `ckpt.pt`: the latest weights, optimizer state and config, used by `--resume`.
  - `best.pt`: the checkpoint with the lowest validation loss, used by `sample.py` by default (`--which latest` picks `ckpt.pt`).
  - `log.csv`: train and validation loss at every evaluation.

### Presets

| preset | for | GPT size | notes |
|---|---|---|---|
| `tiny` | checking the wiring, CPU | ~0.1M params | a few seconds |
| `mac` | laptops and Apple Silicon (MPS), small GPUs | 10.7M params, 256-char context | batch 16 × 4 accumulation steps ≈ 2.2 GB of GPU memory |
| `gpt2-small` | CUDA GPUs with ≥48 GB (H100/H200/A100, RTX 6000-class) | 124M params (GPT-2 small), 1024-token context | bf16, `torch.compile`, fused attention, TinyStories |

Reference speed for the `mac` preset GPT: ~0.65 s/step on an Apple M5 (MacBook Air, 16 GB),
so about 1 hour for 5000 steps. It gets slower if the machine throttles under sustained load,
or if other apps push it into swap. On Apple Silicon the GPU shares system RAM.
For less memory use `--batch_size 8 --grad_accum_steps 8`; the training math is the same.

## Where the data comes from

| dataset | what | size | prepare |
|---|---|---|---|
| TinyShakespeare | Shakespeare plays, character-level (65 chars) | 1.1M chars | `data/shakespeare_char/prepare.py` |
| [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | 2.1M simple stories, GPT-2 BPE tokens | ~470M tokens | `data/tinystories/prepare.py` |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | real educational web text (`sample-10BT`) | 10B tokens | not included yet; the natural next step for big GPUs |

Each prepare script turns the text into token ids once and stores them in
`train.bin`/`val.bin` (flat `uint16`). Training reads random windows from those files
with `np.memmap`, so a 20GB dataset doesn't need 20GB of RAM.

```bash
python data/tinystories/prepare.py --max_stories 5000   # quick subset to try it
python data/tinystories/prepare.py                      # full dataset (~1GB download)
python train.py --model gpt --preset mac --dataset tinystories
```

## NVIDIA GPUs and multi-GPU training

The same code runs on NVIDIA GPUs. The `gpt2-small` preset is a GPT-2-small shape (12 layers,
768 wide, 1024-token context, 124M params) with bf16 mixed precision, `torch.compile` and
fused (Flash) attention.

```bash
# on the GPU server, after cloning:
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python data/tinystories/prepare.py

# 1 GPU
python train.py --model gpt --preset gpt2-small

# 8 GPUs on one machine: torchrun starts 8 processes, one per GPU (DDP)
torchrun --standalone --nproc_per_node=8 train.py --model gpt --preset gpt2-small

# 2 machines x 8 GPUs: run on each node with its own --node_rank (0 and 1)
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=8 \
         --master_addr=<node0-ip> --master_port=29500 train.py --model gpt --preset gpt2-small
```

Tokens per optimizer step = `batch_size × block_size × grad_accum_steps × num_gpus`.
The preset gives 32 × 1024 × 2 × 8 ≈ 0.5M, the value GPT-2 used. With fewer GPUs,
raise `--grad_accum_steps` to keep that number. If you run out of memory, lower
`--batch_size` and raise `--grad_accum_steps` by the same factor; the result is identical.

`--fast_attn` (on in the `mac` and `gpt2-small` presets) uses PyTorch's fused
`scaled_dot_product_attention`. It gives the same results as the explicit attention code.
On CUDA it is much faster and uses much less memory. On MPS it currently brings no gain.

### Testing multi-GPU code without GPUs

DDP also runs on CPU (gloo backend), which is handy for checking the multi-process code path
on a laptop:

```bash
torchrun --standalone --nproc_per_node=2 train.py --model gpt --preset tiny
```

If that hangs on macOS, the host name probably doesn't resolve: Python's `socket.getfqdn()`
can return an unresolvable `…ip6.arpa` name. Give torchrun the address explicitly:

```bash
GLOO_SOCKET_IFNAME=lo0 torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 \
    --master_port=29511 train.py --model gpt --preset tiny
```

## Export to Hugging Face format (for serving)

The `gpt` model has GPT-2's architecture, so `export_hf.py` copies its weights into Hugging
Face's standard `GPT2LMHeadModel`. The script checks that the converted model gives the same
outputs before it saves anything. The result loads in anything that supports GPT-2:
`transformers`, vLLM, TGI, and the serving platforms built on them. Only GPT-2-BPE models
(e.g. `--dataset tinystories`) can be exported; the character-level Shakespeare vocabulary
isn't a standard tokenizer.

```bash
python export_hf.py --ckpt_dir out/tinystories-gpt-gpt2-small --out_dir hf/mini-gpt-tinystories
# -> config.json, generation_config.json, model.safetensors (bf16), tokenizer.json, README.md

# offline server: take the tokenizer from an earlier export instead of downloading it
python export_hf.py --ckpt_dir ... --out_dir ... --tokenizer path/to/earlier/export

# try it with an OpenAI-compatible server
pip install vllm
vllm serve hf/mini-gpt-tinystories --served-model-name mini-gpt
curl localhost:8000/v1/completions -H 'Content-Type: application/json' \
     -d '{"model": "mini-gpt", "prompt": "Once upon a time", "max_tokens": 100}'
```

It's a **base model**: it continues text, so use the completions API (`/v1/completions`). Chat
endpoints need instruction tuning and a chat template, which this model doesn't have.

## Deploy to a model-serving platform

The exported folder is a standard Hugging Face model, so any platform with a vLLM (or TGI)
runtime can serve it. The steps below were written for Red Hat OpenShift AI and VMware Private
AI Services. They are a guide rather than tested recipes: check the details against the
documentation for your platform version.

### 1. Package the model as an OCI image (ModelCar)

Both platforms can pull models from a container registry (Quay, Harbor, ...). A *ModelCar*
is a minimal image that contains only the model files, in `/models`:

```dockerfile
# Containerfile
FROM registry.access.redhat.com/ubi9/ubi-micro:latest
COPY --chown=1001:0 hf/mini-gpt-tinystories /models
USER 1001
```

```bash
podman build -t registry.example.com/models/mini-gpt-tinystories:1.0 .
podman push registry.example.com/models/mini-gpt-tinystories:1.0
```

### 2a. Red Hat OpenShift AI

1. **Model Registry** (dashboard): register the model and a version (e.g. `1.0`) with the
   location `oci://registry.example.com/models/mini-gpt-tinystories:1.0`. Alternatively, upload
   the exported folder to an S3 bucket and register it through an S3 connection.
2. **Deploy:** single-model serving (KServe) with the **vLLM ServingRuntime**, pointing at that
   OCI URI or S3 path. vLLM recognises `GPT2LMHeadModel` from `config.json`.
3. Call the endpoint's `/v1/completions`.

### 2b. VMware Private AI Services (VCF)

1. Push the exported folder to the PAIS **Model Store**, an OCI registry (Harbor), using the
   PAIS CLI (see the PAIS documentation for the exact command and flags).
2. Create a **model endpoint** with the **vLLM** inference engine, pointing at that model.
3. Call its OpenAI-compatible **completions** API.

### Things to know on any platform

- **Use completions, not chat.** This is a base model (see above).
- **The context length is 1024 tokens** (`n_positions` in `config.json`). If the runtime asks
  for a maximum length, set it to 1024 (vLLM: `--max-model-len 1024`).
- **The vocabulary is padded** to 50,304 entries, while the tokenizer has 50,257. Padding to a
  multiple of 64 speeds up GPU matrix maths. The extra ids never occur in the training data, so
  the model learns to give them practically zero probability.
- **The model is small:** 124M parameters, about 250 MB in bf16. It fits easily on one GPU,
  even with many concurrent requests.

## Code map

```
gpt/device.py     cuda/mps/cpu, precision (bf16 on CUDA), DDP setup
gpt/data.py       random (x, y) windows from the .bin files; y = x shifted by one
gpt/tokenizer.py  character tokenizer, GPT-2 BPE (tiktoken)
gpt/bigram.py     model 1
gpt/mlp_lm.py     model 2
gpt/gpt.py        models 3 + 4: attention, feed-forward, transformer block, GPT
gpt/sampling.py   text generation (temperature, top-k)
gpt/config.py     all hyperparameters + presets (tiny / mac / gpt2-small)
train.py          the training loop (AdamW, warmup + cosine LR, grad accumulation, DDP)
sample.py         generate from a checkpoint
export_hf.py      convert a trained GPT to Hugging Face GPT-2 format (safetensors + tokenizer)
notebooks/        explore.ipynb: data, a bigram trained step by step, one training step taken
                  apart (loss, backprop, update), backprop by hand on a tiny network,
                  loss curves, generated text, attention maps
```

## Ideas to try next

- Change one thing at a time and compare `log.csv` curves: `--dropout 0`, `--n_layer 2`, `--block_size 64`.
- Train a GPT on the TinyStories subset, then compare its samples with the Shakespeare ones.
- Add a FineWeb-Edu prepare script (copy the TinyStories one) for a large multi-GPU run.
- Read the classics this code follows: Karpathy's "Let's build GPT" video and nanoGPT;
  "Attention Is All You Need" (2017); the GPT-2 paper (2019).
