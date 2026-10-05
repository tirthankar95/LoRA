# Plan: Running LoRA on macOS (no NVIDIA GPU)

## The core problem

The current code uses **vLLM**, which is built for **NVIDIA CUDA GPUs**. On a Mac
(Apple Silicon M1–M4, or Intel) there is no CUDA device, so:

- vLLM has no usable GPU backend on macOS. Its experimental CPU path is x86-only
  and does not use the Apple GPU (Metal), so it is effectively a non-starter for
  real workloads here.
- Your `configs/config.yaml` now points at a **GGUF** model
  (`Qwen3-8B-GGUF-Q4`). vLLM's GGUF support is partial and still CUDA-oriented —
  it is not the right tool for GGUF on a Mac.

So two things need to change on a Mac: **the inference engine** and (for
training) **the fine-tuning stack**. The good news: Apple Silicon is well
supported by a different ecosystem.

---

## Recommended path: MLX (Apple-native)

[MLX](https://github.com/ml-explore/mlx) is Apple's array framework; `mlx-lm`
runs and fine-tunes LLMs directly on the Apple GPU via Metal. It is the closest
drop-in replacement for what this repo does and supports **both** LoRA training
and inference.

```bash
pip install mlx-lm
```

**Inference:**

```bash
mlx_lm.generate --model Qwen/Qwen3-8B --prompt "Hello"
# or in Python: from mlx_lm import load, generate
```

**LoRA / QLoRA fine-tuning (native, on the Apple GPU):**

```bash
mlx_lm.lora \
  --model Qwen/Qwen3-8B \
  --train \
  --data ./data \
  --iters 1000 \
  --batch-size 4 \
  --num-layers 16        # how many layers get adapters
# then fuse the adapter back into the base weights:
mlx_lm.fuse --model Qwen/Qwen3-8B --adapter-path ./adapters
```

MLX LoRA takes the same conceptual knobs described in
[`docs/architecture.md`](docs/architecture.md) (rank, learning rate, dropout) —
just under MLX's flag/JSON config instead of PEFT's `LoraConfig`.

---

## Alternatives for inference on Mac

| Option | GGUF? | LoRA? | Notes |
|--------|-------|-------|-------|
| **MLX / mlx-lm** | no (uses MLX/safetensors format) | train **and** infer | Best all-round Apple-Silicon choice. Convert HF → MLX with `mlx_lm.convert`. |
| **llama.cpp** | **yes** (native format) | infer with adapter (`--lora`) | C++/Metal, extremely efficient for quantized GGUF. The right engine for your current `Qwen3-8B-GGUF-Q4`. |
| **Ollama** | yes (wraps llama.cpp) | apply adapter via Modelfile | Easiest UX; `ollama run qwen3`. Good for serving, less for training. |
| **HF Transformers + MPS** | no | train + infer (PEFT) | `device="mps"`. Works but slower and more memory-hungry than MLX; fine for small models/datasets. |
| **vLLM** | partial | — | **Not viable on macOS.** Keep it only for a CUDA machine / cloud GPU. |

---

## Alternatives for LoRA *training* on Mac

1. **MLX-LM LoRA (recommended)** — native Metal, QLoRA supported, actively
   maintained for Apple Silicon.
2. **HuggingFace `peft` + `transformers` on MPS** — the same `LoraConfig` API
   this repo already uses; just load the model with `device_map="mps"` (or
   `.to("mps")`) and drop vLLM from training. Slower than MLX but minimal code
   change. Note: bitsandbytes 4-bit QLoRA does **not** run on MPS, so you'd train
   in fp16/bf16 (feasible for ≤~3–8B models on 32GB+ unified memory).
3. **Unsloth** — fast and memory-light, but **CUDA-only**; not an option on Mac.
4. **Cloud GPU** — if you need vLLM/Unsloth/QLoRA-bnb specifically, rent an
   NVIDIA GPU (Colab, Lambda, RunPod, etc.), train there, then run inference
   locally on the Mac with MLX or llama.cpp.

---

## What if I have a GGUF LLM?

GGUF is **llama.cpp's inference/quantization format** — it is designed for *running*
models, not training them. Key facts:

- **You generally do not fine-tune a GGUF file directly.** Train against the
  full-precision (HF safetensors) weights with MLX or PEFT, producing a LoRA
  adapter.
- **llama.cpp *can apply* a LoRA adapter at inference time.** Convert your
  trained adapter to GGUF and pass it in:

  ```bash
  # convert a PEFT/HF LoRA adapter to GGUF
  python llama.cpp/convert_lora_to_gguf.py ./lora_output/adapter --outfile adapter.gguf

  # run the base GGUF model with the adapter applied
  ./llama-cli -m Qwen3-8B-Q4.gguf --lora adapter.gguf -p "your prompt"
  ```

- **To bake the adapter in permanently**, merge it into full-precision weights
  first (PEFT `merge_and_unload()` or `mlx_lm.fuse`), then quantize the merged
  model back to GGUF with `llama.cpp`'s `convert_hf_to_gguf.py` + `llama-quantize`.

### Recommended end-to-end flow for your current setup (Qwen3-8B-GGUF-Q4 on Mac)

```
1. Get the full-precision Qwen3-8B (HF safetensors), not just the GGUF.
2. Fine-tune a LoRA adapter:
     - MLX:  mlx_lm.lora ...            (fastest on Apple Silicon)   ← recommended
     - or:   transformers + peft on MPS (keeps this repo's LoraConfig)
3. Serve on the Mac:
     - MLX:        mlx_lm.fuse, then mlx_lm.generate
     - llama.cpp:  convert adapter → GGUF, run base Q4 GGUF with --lora
4. Keep vLLM only if/when you move to an NVIDIA GPU.
```

---

## Concrete changes to this repo (if you pursue the Mac path)

- **Decouple inference from vLLM.** Make `lora_ft/inference.py` pluggable, with a
  backend chosen in config (e.g. `backend: mlx | llama_cpp | vllm`), so the Mac
  doesn't import vLLM at all.
- **Add an `mlx` training path** (or an MPS `device_map` branch) alongside
  `fine_tune_lora` so training doesn't require CUDA.
- **Fix the config target.** `model_dir` pointing at a GGUF folder won't work for
  PEFT/transformers training (they need HF weights). Separate `base_model` (HF,
  for training) from `gguf_model` (for llama.cpp inference).

> This document is a plan and set of suggestions only — no code has been changed.
> Tell me which path you want (MLX vs. llama.cpp vs. MPS/PEFT) and I'll wire it in.
