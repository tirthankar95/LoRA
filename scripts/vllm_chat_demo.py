"""Standalone demo: apply a chat template and generate with vLLM.

Run from the project root so that the ``lora_ft`` package is importable:
    python -m scripts.vllm_chat_demo
"""
import multiprocessing

try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

from transformers import AutoTokenizer

from lora_ft.inference import generate_with_vllm

# Point these at a local Qwen2-1.5B-Instruct checkout.
MODEL_DIR = "/home/tmittra/models/Qwen2-1.5B-Instruct"

CONVERSATIONS = [
    [{"role": "user", "content": "Tell me a joke"}],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 2?"},
    ],
]


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    for conversation in CONVERSATIONS:
        prompt = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        generate_with_vllm(model_dir=MODEL_DIR, prompt=prompt)
        print("-" * 50)


if __name__ == "__main__":
    main()
