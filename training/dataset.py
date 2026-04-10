# training/dataset.py
# Loads the GSM8K math reasoning dataset from HuggingFace.
# GSM8K contains grade-school math word problems with step-by-step solutions.

from datasets import load_dataset
from typing import Dict


def load_gsm8k(train_size: int = 3000, test_size: int = 1000) -> Dict:
    """
    Load the GSM8K dataset and return train/test subsets.

    Args:
        train_size: Number of training samples to use.
        test_size:  Number of test samples to use.

    Returns:
        Dict with keys 'train' and 'test', each a HuggingFace Dataset slice.
    """
    print("Loading GSM8K dataset from HuggingFace...")
    dataset = load_dataset("gsm8k", "main")

    # Slice to the required sizes (dataset may have more samples)
    train_data = dataset["train"].select(range(min(train_size, len(dataset["train"]))))
    test_data  = dataset["test"].select(range(min(test_size,  len(dataset["test"]))))

    print(f"  Train samples : {len(train_data)}")
    print(f"  Test  samples : {len(test_data)}")

    return {"train": train_data, "test": test_data}


def format_sample(sample: Dict) -> str:
    """
    Format a single GSM8K sample into a prompt string.
    Model input:  'Question: ...'
    Model target: 'Answer: ...'  (the full solution with steps)
    """
    question = sample["question"].strip()
    answer   = sample["answer"].strip()
    return f"Question: {question}\nAnswer: {answer}"
