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
    non_test_size=13000,
    seed=42,
):
    """Prepares the BERT training, validaton, and test data.

    Args:
        tokenize_func: A pre-defined tokenization function.
        dataset_name: The dataset name (identifier) from HuggingFace NLP API.
        train_count: The number of total training data to download.
        train_size: The number of training data used to train the model.
        val_size: The number of validation data used to train the model.
        use_all_test: Whether to use all the test dataset.
        test_count: The number of total test data to download.
        test_size: The number of test data used to test the model.
        others: The number of extra test data.
        seed: The random seed which controls data selection.
    """
    # Load train and test datasets
    train = load_dataset(dataset_name, split="train").shuffle(seed=seed)
    test = load_dataset(dataset_name, split="test").shuffle(seed=seed)

    # Randomly choose train and validation indices
    train_indices, val_indices = train_test_split(
        range(train_count), test_size=val_size, train_size=train_size, random_state=seed
    )
    if not use_all_test:
        _, test_indices = train_test_split(
            range(test_count), test_size=test_size, train_size=non_test_size, random_state=seed
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
    """Prepares a custom dataset in a tsv file.

    Args:
        tokenize_func: A pre-defined tokenization function.
        dataset_name: The dataset name (identifier) from HuggingFace NLP API.
        slice: The number of data points to use.
    """
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
    """Convert the text file into the GPT-2 TextDataset format.

    Args:
        tokenizer: The GPT-2 tokenizer object.
        evaluate: Whether to evalute on the dataset.
    """
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


def text_to_df(filename):
    """Convert a .txt file to a .tsv file, which the data pipeline accepts.

    Args:
        filename: The file path of the .txt file.
    """
    text_file = open(filename, "r").read().split(" <|endoftext|>\n")
    text_file.pop()
    text_file = [i.split("] ") for i in text_file]
    list_file = [[t[1], 1] if "positive" in t[0] else [t[1], 0] for t in text_file]
    assert len(list_file) == len(text_file)
    text = [i[0] for i in list_file]
    label = [i[1] for i in list_file]
    return pd.DataFrame({"text": text, "label": label})
