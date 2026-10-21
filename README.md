# LoRA fine-tuning for **MODELS**

Minimal pipeline to LoRA fine-tune a causal LM, merge the adapter into the base
weights, and query the merged model with [vLLM](https://github.com/vllm-project/vllm).

## Layout

```
.
├── main.py                     # CLI entrypoint (Hydra): train or query
├── configs/
│   └── config.yaml             # model paths + LoRA/training hyperparameters
├── lora_ft/                    # package
│   ├── data.py                 # ASP puzzle training dataset builder
│   ├── train.py                # fine_tune_lora(): train, save adapter + merged model
│   ├── inference.py            # generate_with_vllm(): cached vLLM engine
│   └── utils.py                # get_module_names(): inspect LoRA target modules
├── scripts/
│   └── vllm_chat_demo.py       # standalone chat-template generation demo
├── notebooks/
│   └── gpu_management.ipynb     # VRAM usage exploration
├── data/
│   └── dataset.csv             # ASP puzzle training data (Puzzle and Code columns)
└── docs/
    └── qwen2-1.5B-arch.txt      # dumped module names of the Qwen2 architecture
```

## Setup

```bash
pip install -r requirements.txt
```

Set `model_dir` and `save_dir` in `configs/config.yaml` to local paths.

## Usage

Train a LoRA adapter and save the merged model:

```bash
python main.py action=train
```

Query the merged model:

```bash
python main.py action=query
```

Override any config value on the command line, e.g.
`python main.py train.num_train_epochs=10`.
