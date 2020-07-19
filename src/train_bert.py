import os
import random

import torch
import wandb
from nlp import load_dataset
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import Trainer
from transformers import TrainingArguments


# Enable wandb and watch everything
wandb.login()
os.environ["WANDB_WATCH"] = "all"


class BertTrainer:
    def __init__(
        self,
        model_name_or_path,
        tokenizer_name_or_path,
        dataset_name_or_path,
        training_args,
        model_seed=42,
    ):
        self.model_name_or_path = model_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.dataset_name_or_path = dataset_name_or_path
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name_or_path, use_fast=True
        )
        self.model_seed = model_seed
        self.training_args = training_args

    def tokenize(self, batch):
        """Tokenizes a single batch of text data with padding and truncation."""
        return self.tokenizer(batch["text"], padding=True, truncation=True)

    def prepare_data(self, train_size=0.8, test_size=0.2, seed=42):
        """Prepares training, validation, and test data."""
        train = load_dataset(self.dataset_name_or_path, split="train")
        test = load_dataset(self.dataset_name_or_path, split="test")

        train_indices, val_indices = train_test_split(
            range(len(train)),
            test_size=val_size,
            train_size=train_size,
            random_state=seed,
        )

        train_dataset = train.select(indicies=train_indices)
        val_dataset = val.select(indicies=val_indices)
        test_dataset = test

        train_dataset = train_dataset.map(self.tokenize, batched=True)
        val_dataset = val_dataset.map(self.tokenize, batched=True)
        test_dataset = test_dataset.map(self.tokenize, batched=True)

        train_dataset.set_format(
            "torch", columns["input_ids", "attention_mask", "label"]
        )
        val_dataset.set_format("torch", columns["input_ids", "attention_mask", "label"])
        test_dataset.set_format(
            "torch", columns["input_ids", "attention_mask", "label"]
        )

        return (train_dataset, val_dataset, test_dataset)

    def compute_metrics(self, pred):
        """Computes precision, recall, and F1 score."""
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary"
        )
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    def train_model(self, train_dataset, val_dataset, save_model=True, save_path=None):
        """Trains (fine-tunes) the model."""
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            compute_metrics=self.compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        trainer.train()
        if save_model:
            trainer.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)

    def eval_model(self):
        pass

    
