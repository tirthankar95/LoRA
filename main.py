"""CLI entrypoint: train a LoRA adapter or query the merged model with vLLM.

Usage:
    python main.py                 # uses action from configs/config.yaml
    python main.py action=train
    python main.py action=query
"""
import logging
import multiprocessing
# vLLM/CUDA require the 'spawn' start method; set it before any CUDA work.
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass
from pathlib import Path
import hydra
from omegaconf import DictConfig
from lora_ft.train import fine_tune_lora
from lora_ft.inference import generate_with_vllm

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    if cfg.action == "train":
        fine_tune_lora(cfg)
    elif cfg.action == "query":
        query = input("Enter your query: ")
        merged_dir = Path(cfg.save_dir) / "merged_model"
        if not merged_dir.exists():
            logger.warning(f"No merged model found at {merged_dir}.")
            generate_with_vllm(str(cfg.model_dir), query)
            return
        logger.info(f"Loading model from {merged_dir} with vllm for prompt:\n{query}\n")
        generate_with_vllm(str(merged_dir), query)


if __name__ == "__main__":
    main()
