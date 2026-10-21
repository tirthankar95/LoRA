# Architecture & LoRA Guide

## 1. Code architecture

This repo is a minimal pipeline that LoRA fine-tunes a causal language model,
merges the adapter back into the base weights, and serves the result with vLLM.

### Module map

```
main.py                 Hydra CLI entrypoint. Reads configs/config.yaml and
                        dispatches on `action`:
                          - "train" -> lora_ft.train.fine_tune_lora(cfg)
                          - "query" -> prompt the merged model via vLLM

configs/config.yaml     All runtime knobs: model/save paths, LoRA hyperparams,
                        training hyperparams. Any value is overridable on the
                        CLI, e.g. `python main.py train.num_train_epochs=10`.

lora_ft/                The package.
  data.py               make_train_dataset(tokenizer): builds + tokenizes the
                        ASP puzzle training set from data/dataset.csv.
  train.py              fine_tune_lora(cfg): the core workflow (below).
  inference.py          generate_with_vllm(...): one lazily-created, cached
                        vLLM engine reused across calls.
  utils.py              get_module_names(model_dir): prints every submodule so
                        you can pick valid LoRA `target_modules`.

scripts/vllm_chat_demo.py   Standalone chat-template generation demo.
notebooks/                  VRAM exploration.
data/dataset.csv            ASP puzzle set with Puzzle questions and Code answers.
docs/                       This file + a dump of the Qwen2 module names.
```

### Training data flow (`action=train`)

```
config.yaml
    │
    ▼
AutoTokenizer / AutoModelForCausalLM.from_pretrained(model_dir)
    │
    ▼
make_train_dataset(tokenizer)            # tokenize without padding; mask Puzzle labels
    │
    ▼
get_peft_model(model, LoraConfig)        # inject LoRA adapters into attn proj
    │
    ▼
Trainer(... DataCollatorForAnswerOnly(tokenizer) ...).train()
    │                                    # pad per batch; ignore question and pad
    │                                    # positions in loss with labels = -100
    ▼
save adapter  ->  save_dir/adapter/
merge_and_unload()  ->  save_dir/merged_model/   # standalone weights for vLLM
```

### Query data flow (`action=query`)

```
user prompt ──▶ generate_with_vllm(save_dir/merged_model, prompt)
                     │  (engine cached in module global _ENGINE)
                     ▼
                vLLM SamplingParams(temperature, max_tokens) ──▶ text
```

> **Note:** vLLM targets NVIDIA CUDA GPUs. On an Apple-Silicon Mac this path
> does not work as-is — see [`../plan.md`](../plan.md) for alternatives.

---

## 2. What LoRA is

**LoRA (Low-Rank Adaptation)** fine-tunes a large model by *freezing* all of its
original weights and training a small number of new parameters instead.

For a frozen weight matrix `W` of shape `(d_out, d_in)`, LoRA learns two small
matrices and adds their product as a correction:

```
h = W x  +  (alpha / r) · B A x
            └── frozen ──┘  └─ trainable ─┘

A : (r, d_in)      initialized random (Gaussian)
B : (d_out, r)     initialized to zero  -> the adapter starts as a no-op
r : the rank (r << d_in, d_out)
alpha : scaling factor; the update is scaled by alpha / r
```

Because `r` is tiny (e.g. 8), you train ~0.1–1% of the parameters instead of
100%. Benefits:

- **Memory:** optimizer states exist only for the small A/B matrices, so you can
  fine-tune large models on modest hardware.
- **Modularity:** an adapter is a few MB. You can keep many task-specific
  adapters for one base model and swap them.
- **No inference cost after merge:** `merge_and_unload()` folds `(alpha/r)·BA`
  back into `W`, producing a normal model with zero added latency.

**QLoRA** is LoRA on top of a 4-bit *quantized* frozen base model — the same
trainable adapters, but the frozen weights take ~4× less memory. This is the
usual way to fine-tune 7B–13B models on a single consumer GPU.

---

## 3. Dataset size

LoRA is sample-efficient, but "how much data" depends on the goal:

| Goal | Rough # of examples | Notes |
|------|--------------------|-------|
| Memorize a few facts / style demo | 10 – 100 | Overfit on purpose; high epochs. |
| Teach a narrow skill or format | 500 – 2,000 | A sweet spot for most LoRA tasks. |
| General instruction tuning | 10,000 – 50,000+ | Diversity matters more than raw size. |
| Domain adaptation (new jargon) | 1,000 – 10,000 | Mix in some general data to avoid forgetting. |

Guidelines:

- **Quality > quantity.** A few hundred clean, consistent examples beat thousands
  of noisy ones.
- **Diversity > duplication.** Repeating the same example many times mostly
  teaches the model to memorize, not generalize.
- **Watch for overfitting.** With small sets, hold out ~10% for eval and stop
  when eval loss stops improving. The toy config here (2 samples × 64 epochs) is
  deliberate memorization — *not* a template for real training.
- **Format consistency.** Every example should use the exact prompt template
  you'll use at inference (e.g. the model's chat template).

---

## 4. LoRA configuration — what the values mean

These map to `LoraConfig` (in `lora_ft/train.py`) and `configs/config.yaml`.

| Param | In this repo | Typical range | What it does |
|-------|-------------|---------------|--------------|
| `r` (rank) | 8 | 8 – 64 | Capacity of the adapter. Higher = more expressive but more params / easier to overfit. Start at 8–16; raise to 32–64 for harder tasks or more data. |
| `lora_alpha` | 16 | `r` to `2·r` | Scales the update by `alpha/r`. Common heuristic: `alpha = 2·r`. With `use_rslora=True` the scale becomes `alpha/√r` (more stable at high `r`). |
| `lora_dropout` | 0.05 | 0.0 – 0.1 | Regularization on the adapter input. Use 0.05–0.1 for small datasets, ~0 for large ones. |
| `target_modules` | q,k,v,o proj | attn proj, often + MLP | Which linear layers get adapters. Attention projections are the standard minimum; adding `gate/up/down_proj` increases capacity. |
| `bias` | "none" | "none" | Whether to train bias terms. "none" is almost always fine. |
| `task_type` | CAUSAL_LM | CAUSAL_LM | Decoder-only LM objective (what Qwen/Llama/Mistral use). |
| `use_rslora` | True (code) | True | Rank-Stabilized LoRA; better scaling for larger `r`. |

### Training hyperparameters

| Param | In this repo | Typical range | Notes |
|-------|-------------|---------------|-------|
| `learning_rate` | 2e-4 | 1e-4 – 3e-4 | LoRA tolerates higher LRs than full fine-tuning. 2e-4 is a strong default. |
| `num_train_epochs` | 64 | 1 – 3 (large data), 10 – 50 (tiny data) | 64 is only sane for the 2-sample toy set. For real data, 1–3 passes is typical. |
| `per_device_train_batch_size` | 4 | 1 – 8 | Limited by VRAM. Combine with gradient accumulation for a larger effective batch. |
| (add) `gradient_accumulation_steps` | — | 4 – 16 | Effective batch = batch_size × accum × #GPUs. Raise it when per-device batch must stay small. |
| (add) warmup / scheduler | — | 3% warmup, cosine | Short warmup + cosine decay stabilizes early steps. |

### A sensible real-data starting point

```yaml
lora:
  r: 16
  lora_alpha: 32          # 2 * r
  lora_dropout: 0.05
  bias: "none"
  task_type: "CAUSAL_LM"
train:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  learning_rate: 2e-4
  # gradient_accumulation_steps: 8   # effective batch = 32
```

> Rule of thumb for debugging: if the model won't learn, first raise `r` and
> add MLP layers to `target_modules`; if it overfits, lower `r`/epochs and raise
> `lora_dropout`.
