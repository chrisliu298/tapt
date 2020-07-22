# TAPT: Text Augmentation Using Pre-Trained Transformers

## Features

In this project, we build a text data augmentation pipline based on transformer models and provide source code to set up a proximal policy optimization (PPO) pipeline. The main concept of PPO is optimizing the GPT-2 generate model by using a Bert-like classifer model as supervisor. Therefore, our codes contain 4 parts: Data (prepross) pipline, Classifier, Generator and Metrics (evaluation). 

### 1. Data Pipline

`utils.data_pipeline` provide three methods to pre-process the dataset for training Bert and GPT-2 model.

##### Main methods and Simple Example:

- `prepare_data` to download dataset from Huggingface API

  ```python
  from utils.data_pipeline import prepare_data
  
  # Load Bert dataset
  train_dataset, val_dataset, test_dataset = prepare_data(
    tokenize_func=tokenize,
    dataset_name="yelp_polarity",
    train_count=10,
    train_size=5,
    val_size=5,
    use_all_test=False,
    test_count=10,
    test_size=5,
    others=5,
    seed=42,
  )
  ```

  

- `prepare_custom_data` to process user's data as dataset of training Bert

  ```python
  from utils.data_pipeline import prepare_custom_data
  
  # Load custom data as Bert dataset
  augmented = prepare_custom_data(
    tokenize_func=tokenize, dataset_name="/content/nlp_yelp_train.tsv"
  )
  ```

  

- `get_dataset` to process user's data as dataset of training GPT-2

  ```python
  from utils.data_pipeline import get_dataset
  
  # Load GPT-2 dataset
  train_dataset = (
    get_dataset(data_args, tokenizer=tokenizer) if training_args.do_train else None
   )
  eval_dataset = (
    get_dataset(data_args, tokenizer=tokenizer, evaluate=True）if training_args.do_eval else None
  )
  ```



### 2. Classifier

`classifier` help user to load and use Bert model to classify the sentence.

#### Main method and Simple Example:

- Using the `Classifier` class to load the model and using `classify` to get the result from classifier

  ```python
  from classifier import Classifier
  
  model = AutoModelForSequenceClassification.from_pretrained("/content/drive/My Drive/models/distilroberta_yelp")
  tokenizer = AutoTokenizer.from_pretrained("/content/drive/My Drive/models/distilroberta_yelp", use_fast=True)
  clf = Classifier(model=model, tokenizer=tokenizer)
  
  print(clf.classify("The restaurant is really bad"))
  print(clf.classify("The restaurant is really good"))
  ```



### 3. Generator

`generator` help user to load and use GPT-2 model to generate the sentence depend on given prompt and label.

#### Main method:

- 



#### Simple Example:

## Run the examples

 [Example for Bert classification](src/run_bert_text_classification.py) 

 [Example for GPT-2 generator](src/run_gpt2_language_modeling.py) 

 [Example for PPO pipeline](src/run_gpt2_ppo_language_modeling.py) 


## Members

- Chris Liu

- Jinfan Zhang

- Yiyun Zheng

- Tianyao Zheng