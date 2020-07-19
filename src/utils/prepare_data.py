import pandas as pd
from nlp import load_dataset
from nlp import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer


class DataLoader:
    """A data downloader and preparer.

    Attributes:
        tokenizer: The tokenizer used to preprocess the dataset.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def tokenize(self, batch):
        """Tokenizes a single batch of text data with padding and truncation."""
        return self.tokenizer(batch["text"], padding=True, truncation=True, max_length=512)

    def prepare_data(self, dataset_name_or_path, train_size=0.8, test_size=0.2, seed=42):
        """Prepares training, validation, and test data."""
        train = load_dataset(dataset_name_or_path, split="train")
        test = load_dataset(dataset_name_or_path, split="test")

        train_indices, val_indices = train_test_split(
            range(len(train)),
            test_size=test_size,
            train_size=train_size,
            random_state=seed,
        )

        train_dataset = train.select(indices=train_indices)
        val_dataset = train.select(indices=val_indices)
        test_dataset = test

        train_dataset = train_dataset.map(self.tokenize, batched=True)
        val_dataset = val_dataset.map(self.tokenize, batched=True)
        test_dataset = test_dataset.map(self.tokenize, batched=True)

        train_dataset.set_format(
            "torch", columns=["input_ids", "attention_mask", "label"]
        )
        val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

        return (train_dataset, val_dataset, test_dataset)

    def prepare_custom_data(self, dataset_name_or_path):
        """Prepares a single tsv file dataset."""
        df = pd.read_csv(dataset_name_or_path, delimiter="\t")
        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(self.tokenize, batched=True)
        dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        return dataset
