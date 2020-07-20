from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import TrainingArguments

from train_bert import BertTrainer
from prepare_data import DataLoader


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
# Load model
model = AutoModelForSequenceClassification.from_pretrained("distilroberta-base")

# Create data loader
dataloader = DataLoader(tokenizer)
# Load (original) training, validation, and test set
train_dataset, val_dataset, test_dataset = dataloader.prepare_data(
    "yelp_polarity", train_size=25000, test_size=25000
)
# Load data to evaluate (by training)
augmented = dataloader.prepare_custom_data("/content/gpt2_ppo_yelp_20000.tsv")

# Define training arguments
training_args = TrainingArguments(
    adam_epsilon=1e-08,
    eval_steps=500,
    logging_steps=500,
    evaluate_during_training=True,
    gradient_accumulation_steps=1,
    learning_rate=5e-05,
    logging_dir="/content/logs",
    max_grad_norm=1.0,
    num_train_epochs=4,
    output_dir="/content/drive/My Drive/models/distilroberta_eval",
    per_device_eval_batch_size=32,
    per_device_train_batch_size=32,
    save_steps=500,
    seed=42,
    warmup_steps=0,
    weight_decay=0.0,
)
# Define BERT trainer
bert_trainer = BertTrainer(model=model, tokenizer=tokenizer, training_args=training_args)
# Train BERT
bert_trainer.train_model(augmented, val_dataset, test_dataset, eval_model=True)
