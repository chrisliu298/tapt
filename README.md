# TAPT: Text Augmentation Using Pre-Trained Transformers

## Features

In this project, we build a text data augmentation pipline based on transformer models and provide source code to set up a proximal policy optimization (PPO) pipeline. The main concept of PPO is optimizing the GPT-2 generate model by using a Bert-like classifer model as supervisor. Therefore, our codes contain 4 parts: Data (prepross) pipeline, Classifier, Generator and Metrics (evaluation). 

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

##### Main method and Simple Example:

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

##### Main method and Simple Example:

- `GPT2Generator.generate` using a GPT-2 model to generates a sequence of words of specified length given an input prompt.

  ```python
  import torch
  from pprint import pprint
  from transformers import GPT2LMHeadModel, GPT2Tokenizer
  
  from generator import GPT2Generator
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
  
  gpt2_generator = GPT2Generator(device)
  
  model = GPT2LMHeadModel.from_pretrained("/content/drive/My Drive/models/gpt2_imdb")
  tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
  model.to(device)
  
  prompt = "[positive] <|sep|> The movie is really"
  pprint(gpt2_generator.generate(tokenizer, model, prompt)[0])
  
  prompt = "[negative] <|sep|> The movie is really"
  pprint(gpt2_generator.generate(tokenizer, model, prompt)[0])
  ```

- `GPT2PPOGenerator.generate` using GPT-2 PPO model to generates a sequence of words of specified length given an input prompt.

  ```python
  from trl.gpt2 import GPT2HeadWithValueModel
  from trl.gpt2 import respond_to_batch
  
  from generator import GPT2PPOGenerator
  
  gpt2_ppo_generator = GPT2PPOGenerator(device)
  
  model = GPT2HeadWithValueModel.from_pretrained("/content/drive/My Drive/models/gpt2_ppo_imdb")
  tokenizer = GPT2Tokenizer.from_pretrained("/content/drive/My Drive/models/gpt2_ppo_imdb")
  model.to(device)
  
  prompt = "[positive] The movie is really"
  pprint(gpt2_ppo_generator.generate(tokenizer, model, prompt))
  
  prompt = "[negative] The movie is really"
  pprint(gpt2_ppo_generator.generate(tokenizer, model, prompt))
  ```

#### 

### 4. Metrics

`utils.data_pipeline` provide two methods to help users to evaluate the BERT model's accuracy and to evaluate the GPT-2 model's perplexity.

##### Main method and Simple Example:

- `compute_metrics` to compute precision, recall, and F1 score of BERT model.

  ```python
  from transformers import Trainer
  from utils.metrics import compute_metrics
  
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
  ```

- `evaluate_gpt2` to computes the perplexity score of GPT-2.

  ```python
      from utils.metrics import evaluate_gpt2
    
    	# Evaluate the model
      ppl = evaluate_gpt2("/content/test.txt", training_args, data_args, trainer, tokenizer)
      print(ppl)
  ```

## Run the examples

We also provide three sample scripts to help user set up their training pipeline with our modules and Huggingface API.

 [Example for Bert classification](src/run_bert_text_classification.py) 

 [Example for GPT-2 generator](src/run_gpt2_language_modeling.py) 

 [Example for PPO pipeline](src/run_gpt2_ppo_language_modeling.py) 


## Members

- Chris Liu
- Yiyun Zheng
- Jinfan Zhang
- Tianyao Zheng
