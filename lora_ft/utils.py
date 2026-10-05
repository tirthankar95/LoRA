"""Miscellaneous helpers for inspecting models."""
from transformers import AutoModelForCausalLM


def get_module_names(model_dir: str):
    """Print the name and class of every submodule in a model.

    Useful for discovering valid ``target_modules`` for a LoRA config.
    """
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    for name, module in model.named_modules():
        print(name, module.__class__.__name__)
