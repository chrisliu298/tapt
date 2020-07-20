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
        return self.tokenizer(
            batch["text"], padding="max_length", truncation=True, max_length=512
        )

    def prepare_data(
        self,
        dataset_name_or_path,
        train_split=0.8,
        val_split=0.2,
        seed=42,
        train_size=None,
        test_size=None,
    ):
        """Prepares training, validation, and test data."""
        if train_size and test_size:
            train = load_dataset(dataset_name_or_path, split=f"train[:{train_size}]")
            test = load_dataset(dataset_name_or_path, split=f"test[:{test_size}]")
        else:
            train = load_dataset(dataset_name_or_path, split="train")
            test = load_dataset(dataset_name_or_path, split="test")

        train_indices, val_indices = train_test_split(
            range(len(train)),
            test_size=val_split,
            train_size=train_split,
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
        val_dataset.set_format(
            "torch", columns=["input_ids", "attention_mask", "label"]
        )
        test_dataset.set_format(
            "torch", columns=["input_ids", "attention_mask", "label"]
        )

        return (train_dataset, val_dataset, test_dataset)

    def prepare_custom_data(self, dataset_name_or_path, slice=None):
        """Prepares a single tsv file dataset."""
        df = pd.read_csv(dataset_name_or_path, delimiter="\t").sample(frac=1)
        if slice:
            start = int(slice[1:-1].split(":")[0])
            stop = int(slice[1:-1].split(":")[1])
            df = df[start:stop]
        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(self.tokenize, batched=True)
        dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        return dataset
