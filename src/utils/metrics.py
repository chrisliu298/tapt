from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support


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
