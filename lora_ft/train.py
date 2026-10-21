"""LoRA fine-tuning of a causal LM and saving the merged weights for vLLM."""
import logging
from pathlib import Path

import torch
from omegaconf import DictConfig
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from lora_ft.data import DataCollatorForAnswerOnly, make_train_dataset

logger = logging.getLogger(__name__)


def fine_tune_lora(cfg: DictConfig) -> str:
    """Fine-tune a LoRA adapter and (optionally) save the merged model.

    Returns the path to the saved adapter directory.

    Training-step arithmetic:
        steps_per_epoch = ceil(len(train_dataset) / per_device_train_batch_size)
        total_steps     = steps_per_epoch * num_train_epochs
    (``per_device_train_batch_size`` is the batch size per GPU.)
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_dir)
    train_dataset = make_train_dataset(tokenizer)
    lora_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        use_rslora=True,
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        task_type=cfg.lora.task_type,
    )
    peft_model = get_peft_model(model, lora_config)
    # Pad only to the longest sequence in each batch. Prompt and padding tokens
    # are already marked -100 in labels and therefore do not contribute to loss.
    data_collator = DataCollatorForAnswerOnly(tokenizer)
    training_args = TrainingArguments(
        output_dir=cfg.train.output_dir,
        num_train_epochs=cfg.train.num_train_epochs,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        logging_steps=cfg.train.logging_steps,
        save_strategy="no",  # the PEFT adapter is saved separately below
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    # Save the PEFT adapter.
    adapter_dir = Path(cfg.save_dir) / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(adapter_dir))
    logger.info(f"Saved LoRA adapter to {adapter_dir}")
    # Merge the adapter into the base weights and save a standalone model that
    # vLLM can load directly.
    merged_dir = Path(cfg.save_dir) / "merged_model"
    if cfg.lora.merge_and_save or not merged_dir.exists():
        try:
            merged = peft_model.merge_and_unload()
            merged_dir.mkdir(parents=True, exist_ok=True)
            merged.save_pretrained(str(merged_dir))
            tokenizer.save_pretrained(str(merged_dir))
            logger.info(f"Merged model saved to {merged_dir}")
        except Exception:
            logger.warning(
                "merge_and_unload() not available for this PEFT version. "
                "Saved adapter only."
            )
    return str(adapter_dir)