import collections
import math
import os

from dict2obj import Dict2Obj
from transformers import AutoConfig
from transformers import AutoModelWithLMHead
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer
from transformers import TrainingArguments
from transformers import set_seed

from utils.data_pipeline import get_dataset
from utils.metrics import evaluate_gpt2


START_TOKEN = "<|startoftext|>"
SEP_TOKEN = "<|sep|>"


def main():
    # Model arguments
    model_args = collections.defaultdict(
        config_name="gpt2",
        model_name_or_path="gpt2-medium",
        model_type="gpt2",
        tokenizer_name="gpt2",
        cache_dir=None,
    )

    # Data arguments
    data_args = collections.defaultdict(
        train_data_file="/content/train.txt",
        eval_data_file="/content/val.txt",
        line_by_line=False,
        mlm=False,
        mlm_probability=0.15,
        block_size=512,
        overwrite_cache=False,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="/content",
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        do_predict=False,
        evaluate_during_training=True,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        weight_decay=0.0,
        adam_epsilon=1e-08,
        max_grad_norm=1.0,
        num_train_epochs=5,
        max_steps=-1,
        warmup_steps=0,
        logging_dir=None,
        logging_first_step=False,
        logging_steps=1000,
        eval_steps=1000,
        save_steps=2000,
        save_total_limit=100000,
        no_cuda=False,
        seed=42,
        fp16=False,
        fp16_opt_level="O1",
        local_rank=-1,
    )

    # Convert dict to objects
    model_args = Dict2Obj(model_args)
    data_args = Dict2Obj(data_args)

    # Sed seed
    set_seed(training_args.seed)

    # Load tokenizer and model
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir
    )
    model = AutoModelWithLMHead.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Add special tokens
    tokenizer.add_special_tokens({"sep_token": SEP_TOKEN})
    tokenizer.add_special_tokens({"bos_token": START_TOKEN})
    model.resize_token_embeddings(len(tokenizer))

    # Load dataset
    train_dataset = (
        get_dataset(data_args, tokenizer=tokenizer) if training_args.do_train else None
    )
    eval_dataset = (
        get_dataset(data_args, tokenizer=tokenizer, evaluate=True)
        if training_args.do_eval
        else None
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=data_args.mlm,
        mlm_probability=data_args.mlm_probability,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=True,
    )

    # Define model path
    model_path = (
        model_args.model_name_or_path
        if model_args.model_name_or_path is not None
        and os.path.isdir(model_args.model_name_or_path)
        else None
    )

    # Train the model
    # train_results = trainer.train(model_path=model_path)
    # trainer.save_model()
    # tokenizer.save_pretrained(training_args.output_dir)

    # Evaluate the model
    ppl = evaluate_gpt2(
        "/content/test.txt", training_args, data_args, trainer, tokenizer
    )
    print(ppl)


if __name__ == "__main__":
    main()
