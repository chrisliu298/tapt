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

    def compute_metrics(self, pred):
        """Computes precision, recall, and F1 score."""
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary"
        )
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    def train_model(
        self,
        train_dataset,
        val_dataset,
        test_dataset,
        eval_model=False,
        save_model=False,
        save_path=None,
    ):
        """Trains (fine-tunes) and evaluates the model."""
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            compute_metrics=self.compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        trainer.train()

        if eval_model:
            train_score = trainer.evaluate(eval_dataset=train_dataset)
            valscore = trainer.evaluate(eval_dataset=valdataset)
            test_score = trainer.evaluate(eval_dataset=test_dataset)

        if save_model:
            trainer.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
