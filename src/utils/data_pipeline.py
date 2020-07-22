import pandas as pd
from nlp import load_dataset
from nlp import Dataset
from sklearn.model_selection import train_test_split
from transformers import LineByLineTextDataset
from transformers import TextDataset


def prepare_data(
    tokenize_func,
    dataset_name,
    train_count,
    train_size,
    val_size,
    use_all_test=False,
    test_count=38000,
    test_size=25000,
    others=13000,
    seed=42,
):
    # Load train and test datasets
    train = load_dataset(dataset_name, split="train").shuffle(seed=seed)
    test = load_dataset(dataset_name, split="test").shuffle(seed=seed)

    # Randomly choose train and validation indices
    train_indices, val_indices = train_test_split(
        range(train_count), test_size=val_size, train_size=train_size, random_state=seed
    )
    if not use_all_test:
        _, test_indices = train_test_split(
            range(test_count), test_size=test_size, train_size=others, random_state=seed
        )

    # Split train and validation data
    train_dataset = train.select(indices=train_indices)
    val_dataset = train.select(indices=val_indices)
    test_dataset = test.select(indices=test_indices) if not use_all_test else test
    # Preprocess
    train_dataset = train_dataset.map(tokenize_func, batched=True)
    val_dataset = val_dataset.map(tokenize_func, batched=True)
    test_dataset = test_dataset.map(tokenize_func, batched=True)
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    return (train_dataset, val_dataset, test_dataset)


def prepare_custom_data(tokenize_func, dataset_name, slice=None):
    df = pd.read_csv(dataset_name, delimiter="\t").sample(frac=1)
    if slice:
        start = int(slice[1:-1].split(":")[0])
        stop = int(slice[1:-1].split(":")[1])
        df = df[start:stop]
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(tokenize_func, batched=True)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return dataset


def get_dataset(args, tokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        return LineByLineTextDataset(
            tokenizer=tokenizer, file_path=file_path, block_size=args.block_size
        )
    else:
        return TextDataset(
            tokenizer=tokenizer,
            file_path=file_path,
            block_size=args.block_size,
            overwrite_cache=args.overwrite_cache,
        )
