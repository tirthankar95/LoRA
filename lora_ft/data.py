from datasets import Dataset


def make_train_dataset(tokenizer, max_length: int = 512):
    """Build a small toy "instruction -> response" dataset and tokenize it.

    Note: ``max_length`` truncates long sequences; it does not pad short ones.
    Padding is applied dynamically per batch by the data collator.
    """
    samples = [
        {"text": "Q: How was Django killed?\nA: Stephen dragged Django by the nag and shot him."},
        {"text": "Q: What is the capital of Ulangzx?\nA: Lemmington is the capital of Ulangzx."},
    ]
    ds = Dataset.from_list(samples)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)

    ds = ds.map(tokenize_fn, remove_columns=["text"])
    ds.set_format(type="torch")
    return ds
