import pandas as pd
from ppinrt import pprint
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import Trainer
from transformers import TrainingArguments

from utils.data_pipeline import prepare_data
from utils.data_pipeline import prepare_custom_data
from utils.metrics import compute_metrics


def tokenize(batch):
    """Tokenize a batch of data (with padding and truncation).

    Arg:
        batch: A batch of training data.
    """
    return tokenizer(
        batch["text"], padding="max_length", truncation=True, max_length=512
    )


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
# Load model
model = AutoModelForSequenceClassification.from_pretrained("distilroberta-base")

# Load dataset
train_dataset, val_dataset, test_dataset = prepare_data(
    tokenize_func=tokenize,
    dataset_name="yelp_polarity",
    train_count=10000,
    train_size=5000,
    val_size=5000,
    use_all_test=False,
    test_count=38000,
    test_size=25000,
    others=13000,
    seed=42,
)
# Load custom data
augmented = prepare_custom_data(
    tokenize_func=tokenize, dataset_name="/content/gpt2_ppo_yelp_5000.tsv"
)

# Define training arguments
training_args = TrainingArguments(
    adam_epsilon=1e-08,
    eval_steps=1000,
    logging_steps=1000,
    evaluate_during_training=True,
    gradient_accumulation_steps=1,
    learning_rate=5e-05,
    logging_dir="/content/logs",
    max_grad_norm=1.0,
    num_train_epochs=4,
    output_dir="/content/drive/My Drive/models/distilroberta",
    per_device_eval_batch_size=32,
    per_device_train_batch_size=32,
    save_steps=1000,
    seed=42,
    warmup_steps=0,
    weight_decay=0.0,
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=augmented,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Evaluate the model
train_score = trainer.evaluate(eval_dataset=train_dataset)
val_score = trainer.evaluate(eval_dataset=val_dataset)
test_score = trainer.evaluate(eval_dataset=test_dataset)

pprint(train_score)
pprint(val_score)
pprint(test_score)
