"""vLLM-backed text generation.

A single, lazily-initialized vLLM engine is reused across calls so the model
weights are only loaded into GPU memory once per process.
"""
import logging

try:
    from vllm import LLM, SamplingParams
except Exception:  # vllm is an optional, GPU-only dependency
    LLM = None
    SamplingParams = None

logger = logging.getLogger(__name__)

_ENGINE = None


def _get_engine(model_dir, tokenizer=None):
    global _ENGINE
    if LLM is None:
        raise RuntimeError("vllm is not installed or failed to import.")
    if _ENGINE is None:
        kwargs = dict(
            model=model_dir,
            max_model_len=8192,
            max_num_batched_tokens=8192,
            gpu_memory_utilization=0.8,
            dtype="float16",
        )
        if tokenizer is not None:
            kwargs["tokenizer"] = tokenizer
        _ENGINE = LLM(**kwargs)
    return _ENGINE


def generate_with_vllm(model_dir, prompt, tokenizer=None, temperature=0.7, max_tokens=8192):
    """Generate text for ``prompt`` using vLLM, reusing a cached engine."""
    engine = _get_engine(model_dir, tokenizer=tokenizer)
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature)
    outputs = engine.generate(prompt, sampling_params=sampling_params)
    for out in outputs:
        logger.info("=== vllm generation ===")
        logger.info(out.outputs[0].text)
    return outputs
