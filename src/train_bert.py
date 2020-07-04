from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    Trainer,
    TrainingArguments,
)
from nlp import load_dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split


def compute_metrics(pred):
    """Compute precision, recall, and F1 score.

    Arg:
        pred: The model prediction.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def main():
    """Fine-tune BERT model.
    """

    def tokenize(batch):
        """Tokenize a batch of data (with padding and truncation).

        Arg:
            batch: A batch of training data.
        """
        return tokenizer(batch["text"], padding=True, truncation=True,)

    # Load model and tokenizer
    model = BertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # Load train and test datasets
    imdb_train = load_dataset("imdb", split="train")
    imdb_test = load_dataset("imdb", split="test")

    # Randomly choose train and validation indices
    train_indices, val_indices = train_test_split(
        range(len(imdb_train)), test_size=0.2, train_size=0.8
    )
    # Split train and validation data
    train_dataset = imdb_train.select(indices=train_indices)
    val_dataset = imdb_train.select(indices=val_indices)
    test_dataset = imdb_test

    # Preprocess
    train_dataset = train_dataset.map(
        tokenize, batched=True, batch_size=len(train_dataset)
    )
    val_dataset = val_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
    test_dataset = test_dataset.map(
        tokenize, batched=True, batch_size=len(train_dataset)
    )
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        evaluate_during_training=True,
        logging_dir="./logs",
    )

    # Define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Fine-tune the model
    trainer.train()

    # Evaluate the model
    trainer.evaluate(eval_dataset=test_dataset)


if __name__ == "__main__":
    main()
