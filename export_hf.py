"""
Export a trained GPT checkpoint to Hugging Face format (GPT2LMHeadModel).

Our GPT (gpt/gpt.py) has the same architecture as OpenAI's GPT-2: pre-LayerNorm blocks,
learned position embeddings, tied input/output embeddings. So instead of shipping custom
model code, we copy the weights into Hugging Face's standard GPT-2 class. Everything that
understands GPT-2 can then load it: transformers, vLLM, TGI, and model-serving platforms
built on them (Red Hat OpenShift AI, VMware Private AI Services, ...).

Output folder (what every HF-compatible tool expects):
    config.json              architecture + sizes
    generation_config.json   default sampling settings
    model.safetensors        the weights (safe, memory-mappable format)
    tokenizer.json, ...      GPT-2 BPE tokenizer (same as tiktoken "gpt2" used in training)
    tokenizer_config.json    tokenizer settings + chat template (see --chat_template)
    README.md                short model card

Usage:
    python export_hf.py --ckpt_dir out/tinystories-gpt-gpt2-small --out_dir hf/mini-gpt-tinystories

Chat template: this is a base model (it continues text), but chat APIs and playgrounds
(/v1/chat/completions) need a template that turns a list of messages into one prompt.
By default a "continue" template is added: the conversation is joined into one running
text, so each reply continues the story. --chat_template none leaves it out, or pass a
path to your own .jinja template.

Only `--model gpt` checkpoints trained on a GPT-2 BPE dataset (e.g. tinystories) can be
exported this way. The character-level Shakespeare models use a custom 65-character
vocabulary that standard tokenizers and serving runtimes don't know.

Requires: pip install transformers
"""

import argparse
import json
import os

import torch

from sample import load_model

# How our parameter names map to Hugging Face GPT-2 names.
# Linear layers: our nn.Linear stores weight as (out, in); HF GPT-2 uses Conv1D with
# (in, out) -> transpose. Our linear layers have no bias; GPT-2's do -> set them to 0.
LINEAR_MAP = {  # ours (inside block i)    -> HF (inside transformer.h.i)
    "attn.qkv": "attn.c_attn",            # q, k, v projections, same q|k|v order
    "attn.proj": "attn.c_proj",           # attention output projection
    "ffwd.net.0": "mlp.c_fc",             # feed-forward expand  C -> 4C
    "ffwd.net.2": "mlp.c_proj",           # feed-forward project 4C -> C
}
NORM_MAP = {"ln1": "ln_1", "ln2": "ln_2"}  # LayerNorms: weight and bias copied as-is

# Chat template for a base model: join all non-system messages into one running text, so the
# model's reply simply continues it. (Instruction-tuned models need role markers instead.)
CHAT_TEMPLATES = {
    "continue": "{%- for message in messages -%}{%- if message['role'] != 'system' -%}"
                "{{ message['content'] }}{%- endif -%}{%- endfor -%}",
}


def to_hf(model, cfg, vocab_size):
    """Build a GPT2LMHeadModel with exactly our weights."""
    from transformers import GPT2Config, GPT2LMHeadModel

    hf_cfg = GPT2Config(
        vocab_size=vocab_size,            # padded vocab (e.g. 50304); ids >= 50257 are never produced in practice
        n_positions=cfg.block_size,       # max context length
        n_embd=cfg.n_embd, n_layer=cfg.n_layer, n_head=cfg.n_head,
        n_inner=4 * cfg.n_embd,
        activation_function="gelu",       # exact GELU, as nn.GELU() (GPT-2 itself used the tanh approximation)
        layer_norm_epsilon=1e-5,          # nn.LayerNorm default
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,  # inference only
        bos_token_id=50256, eos_token_id=50256,           # <|endoftext|>
        tie_word_embeddings=True,
    )
    hf = GPT2LMHeadModel(hf_cfg).eval()
    ours = model.state_dict()
    target = hf.state_dict()
    assigned = set()

    def put(hf_name, tensor):
        assert target[hf_name].shape == tensor.shape, (hf_name, target[hf_name].shape, tensor.shape)
        target[hf_name].copy_(tensor)
        assigned.add(hf_name)

    with torch.no_grad():
        put("transformer.wte.weight", ours["token_embedding.weight"])
        put("transformer.wpe.weight", ours["position_embedding.weight"])
        for i in range(cfg.n_layer):
            for o, h in NORM_MAP.items():
                put(f"transformer.h.{i}.{h}.weight", ours[f"blocks.{i}.{o}.weight"])
                put(f"transformer.h.{i}.{h}.bias", ours[f"blocks.{i}.{o}.bias"])
            for o, h in LINEAR_MAP.items():
                w = ours[f"blocks.{i}.{o}.weight"]                       # (out, in)
                put(f"transformer.h.{i}.{h}.weight", w.t().contiguous())  # (in, out)
                put(f"transformer.h.{i}.{h}.bias", torch.zeros(w.shape[0]))
        put("transformer.ln_f.weight", ours["ln_f.weight"])
        put("transformer.ln_f.bias", ours["ln_f.bias"])
        assigned.add("lm_head.weight")  # tied to transformer.wte.weight

    missing = set(target) - assigned
    assert not missing, f"HF parameters not filled from the checkpoint: {sorted(missing)}"
    return hf


@torch.no_grad()
def check_same_outputs(model, hf, vocab_size, block_size):
    """Both models must produce the same next-token scores for the same input."""
    ids = torch.randint(0, min(vocab_size, 50257), (2, min(64, block_size)))
    ours, _ = model(ids)
    theirs = hf(ids).logits
    diff = (ours - theirs).abs().max().item()
    print(f"max |logits difference| ours vs HF: {diff:.2e}")
    assert diff < 1e-3, "converted model does not match - not saving"


def model_card(cfg, ckpt, name, has_chat_template=False):
    loss = ckpt.get("val_loss", ckpt["best_val"])
    params = (12 * cfg.n_layer * cfg.n_embd ** 2) / 1e6
    chat_section = ("""## Chat API

There is no instruction tuning, but the tokenizer has a simple chat template so chat APIs and
playgrounds work: all non-system messages are joined into one running text and the reply
continues it. Write the beginning of a story as the user message; don't ask it questions.

""" if has_chat_template else "")
    return f"""---
library_name: transformers
pipeline_tag: text-generation
---

# {name}

A small GPT-2-architecture language model trained from scratch with the mini-GPT project
(PyTorch). It is a **base model** (no instruction or chat tuning): give it the beginning of a
text and it continues it. The completions API is the natural way to call it.

| | |
|---|---|
| architecture | GPT-2 (`GPT2LMHeadModel`), {cfg.n_layer} layers, {cfg.n_head} heads, {cfg.n_embd} wide, ~{params:.0f}M non-embedding params |
| context length | {cfg.block_size} tokens |
| tokenizer | GPT-2 BPE (50,257 tokens) |
| training data | `{cfg.dataset}` |
| training | {ckpt['iter']} steps, validation loss {loss:.3f} |

## Usage

```python
from transformers import pipeline
generate = pipeline("text-generation", model="path/to/this/folder")
print(generate("Once upon a time", max_new_tokens=100)[0]["generated_text"])
```

```bash
vllm serve path/to/this/folder --served-model-name {name}
curl localhost:8000/v1/completions -H 'Content-Type: application/json' \\
     -d '{{"model": "{name}", "prompt": "Once upon a time", "max_tokens": 100}}'
```

{chat_section}## Limitations

Small model trained on a single narrow dataset: expect fluent text in the style of the
training data, not facts or reasoning. Not suitable for production use cases beyond demos
and experiments.
"""


def add_chat_template(out_dir: str, spec: str):
    """
    Write the chat template into tokenizer_config.json ("chat_template" key). That location is
    read by every transformers version, old and new, so it works whatever version the serving
    runtime bundles. Returns the template, or None for --chat_template none.
    """
    if spec == "none":
        return None
    template = CHAT_TEMPLATES[spec] if spec in CHAT_TEMPLATES else open(spec).read()
    path = os.path.join(out_dir, "tokenizer_config.json")
    with open(path) as f:
        tok_cfg = json.load(f)
    tok_cfg["chat_template"] = template
    with open(path, "w") as f:
        json.dump(tok_cfg, f, indent=2, ensure_ascii=False)
    # newer transformers can also keep the template in a separate file; keep a single source
    jinja = os.path.join(out_dir, "chat_template.jinja")
    if os.path.exists(jinja):
        os.remove(jinja)
    return template


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt_dir", required=True, help="training output folder, e.g. out/tinystories-gpt-gpt2-small")
    p.add_argument("--out_dir", required=True, help="where to write the Hugging Face model folder")
    p.add_argument("--which", default="best", choices=["best", "latest"], help="best.pt or ckpt.pt")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                   help="weight precision on disk (bf16 = half the size, what GPUs serve in anyway)")
    p.add_argument("--name", default=None, help="model name used in the model card (default: out_dir name)")
    p.add_argument("--chat_template", default="continue",
                   help="chat template for /v1/chat/completions: 'continue' (default: the conversation "
                        "is joined into one text the model continues), 'none', or a path to a .jinja file")
    p.add_argument("--tokenizer", default="openai-community/gpt2",
                   help="where to get the GPT-2 tokenizer files: a Hugging Face repo id (needs internet once) "
                        "or a local folder with tokenizer.json, e.g. from a previous export (offline servers)")
    args = p.parse_args()

    from transformers import AutoTokenizer, GenerationConfig

    # 1. load our checkpoint (on CPU - no GPU needed for converting)
    model, cfg, ckpt = load_model(args.ckpt_dir, "cpu", args.which)
    if cfg.model != "gpt":
        raise SystemExit(f"only --model gpt can be exported to GPT-2 format, this checkpoint is {cfg.model!r}")
    meta_tokenizer = __import__("pickle").load(open(os.path.join("data", cfg.dataset, "meta.pkl"), "rb"))["tokenizer"]
    if meta_tokenizer != "gpt2":
        raise SystemExit(f"dataset {cfg.dataset!r} uses the {meta_tokenizer!r} tokenizer; "
                         "only GPT-2 BPE models (e.g. --dataset tinystories) can be exported")
    for m in model.modules():          # explicit attention path: plain math, nothing fused
        if hasattr(m, "fast"):
            m.fast = False
    vocab_size = ckpt["vocab_size"]
    print(f"loaded {cfg.model} from {args.ckpt_dir} (step {ckpt['iter']}), "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, vocab {vocab_size}")

    # 2. copy weights into Hugging Face GPT-2 and prove the outputs are identical
    hf = to_hf(model, cfg, vocab_size)
    check_same_outputs(model, hf, vocab_size, cfg.block_size)

    # 3. save model (safetensors), generation defaults, tokenizer, model card
    os.makedirs(args.out_dir, exist_ok=True)
    hf.to(getattr(torch, args.dtype)).save_pretrained(args.out_dir, safe_serialization=True)
    GenerationConfig(bos_token_id=50256, eos_token_id=50256, pad_token_id=50256,
                     do_sample=True, temperature=0.8, top_k=50, max_new_tokens=200).save_pretrained(args.out_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)  # repo id (downloads ~4MB once) or local folder
    tokenizer.model_max_length = cfg.block_size
    tokenizer.save_pretrained(args.out_dir)
    chat_template = add_chat_template(args.out_dir, args.chat_template)
    name = args.name or os.path.basename(os.path.normpath(args.out_dir))
    with open(os.path.join(args.out_dir, "README.md"), "w") as f:
        f.write(model_card(cfg, ckpt, name, chat_template is not None))

    print(f"saved Hugging Face model to {args.out_dir}:")
    for fname in sorted(os.listdir(args.out_dir)):
        print(f"  {fname:28} {os.path.getsize(os.path.join(args.out_dir, fname)) / 1e6:8.1f} MB")


if __name__ == "__main__":
    main()
