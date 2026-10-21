from datasets import Dataset
from pathlib import Path
import pandas as pd
import logging
import torch

logger = logging.getLogger(__name__)


def dummy_text():
    return [
        {"text": "Q: How was Django killed?\nA: Stephen dragged Django by the nag and shot him."},
        {"text": "Q: What is the capital of Ulangzx?\nA: Lemmington is the capital of Ulangzx."},
    ]


def asp_puzzle():
    path = Path(__file__).resolve().parent.parent / "data" / "dataset.csv"
    df = pd.read_csv(path)
    required_columns = {"Puzzle", "Code"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"{path} is missing required columns: {', '.join(sorted(missing_columns))}"
        )
    if df.empty:
        raise ValueError(f"No ASP puzzle training examples found in {path}.")

    common_puzzle_suffix = (
        "\nSolve the grid puzzle directly or indirectly by writing some code in ASP, that gives the solution when executed."
    )
    samples = []
    rows = df[["Puzzle", "Code"]].itertuples(index=False)
    for row_number, row in enumerate(rows, start=1):
        question, answer = row
        if pd.isna(question) or pd.isna(answer):
            logger.error(
                f"Missing Puzzle or Code value in {path} at CSV row {row_number}."
            )
            continue
        samples.append(
            {
                "question": f"{question}{common_puzzle_suffix}\n",
                "answer": str(answer),
            }
        )
    return samples


def make_train_dataset(tokenizer):
    """Build tokenized examples with loss enabled only for answer tokens."""
    samples = asp_puzzle()
    ds = Dataset.from_list(samples)
    def tokenize_fn(examples):
        questions = tokenizer(examples["question"], add_special_tokens=True, \
                              truncation=False)
        answers = tokenizer(examples["answer"], add_special_tokens=False, \
                            truncation=False)
        input_ids, labels = [], []
        for question_ids, answer_ids in zip(
            questions["input_ids"],
            answers["input_ids"],
        ):
            if tokenizer.eos_token_id is not None and (
                not answer_ids or answer_ids[-1] != tokenizer.eos_token_id
            ):
                answer_ids = answer_ids + [tokenizer.eos_token_id]
            sequence_ids = question_ids + answer_ids
            input_ids.append(sequence_ids)
            labels.append([-100] * len(question_ids) + answer_ids)
        return {"input_ids": input_ids, "labels": labels}
    ds = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)
    ds.set_format(type="torch")
    return ds


class DataCollatorForAnswerOnly:
    """Right-pad each batch and ignore prompt and padding tokens in loss."""

    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id
        if self.pad_token_id is None:
            raise ValueError("Tokenizer must define a pad_token_id or eos_token_id.")

    def __call__(self, features):
        if not features:
            raise ValueError("Cannot collate an empty batch.")
        sequences = [feature["input_ids"] for feature in features]
        targets = [feature["labels"] for feature in features]
        lengths = [len(sequence) for sequence in sequences]
        max_length = max(lengths)
        input_ids = torch.full(
            (len(features), max_length), self.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((len(features), max_length), dtype=torch.long)
        labels = torch.full(
            (len(features), max_length), -100,     
            dtype=torch.long
        )
        for row, (sequence, target, length) in enumerate(
            zip(sequences, targets, lengths)
        ):
            if len(target) != length:
                raise ValueError(
                    "Each example's input_ids and labels must have equal lengths."
                )
            start = 0
            end = start + length
            input_ids[row, start:end] = torch.as_tensor(sequence, dtype=torch.long)
            attention_mask[row, start:end] = 1
            labels[row, start:end] = torch.as_tensor(target, dtype=torch.long)
        '''
        input_ids: tell the model what tokens it receives.
        attention_mask: tells it which positions are real tokens versus padding.
        labels: determine which token predictions contribute to the loss.
        '''
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
