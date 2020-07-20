from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

from utils import get_dataset


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


def evaluate_gpt2(data, training_args, data_args, trainer, tokenizer):
    training_args.do_eval = True
    data_args.eval_data_file = data
    eval_dataset = get_dataset(data_args, tokenizer=tokenizer, evaluate=True) if training_args.do_eval else None
    trainer.eval_dataset = eval_dataset
    eval_output = trainer.evaluate()
    eval_perplexity = math.exp(eval_output["eval_loss"])
    return {"perplexity": eval_perplexity}