import os
import random

import torch

# import wandb
from nlp import load_dataset
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import Trainer
from transformers import TrainingArguments


# Enable wandb and watch everything
# wandb.login()
# os.environ["WANDB_WATCH"] = "all"


class BertTrainer:
    """A trainer of BERT models.

    Attributes:
        model: The bert model object.
        tokenizer: The tokenizer object.
        training_args: The arguments of trainign the BERT model.
    """

    def __init__(
        self, model, tokenizer, training_args,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.training_args = training_args

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
        self, train_dataset, val_dataset, test_dataset, eval_model=False,
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
            val_score = trainer.evaluate(eval_dataset=val_dataset)
            test_score = trainer.evaluate(eval_dataset=test_dataset)
            return (train_score, val_score, test_score)
