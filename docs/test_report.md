# System and Unit Test Report

Product name: Text Augmentation Using Pre-Trained Transformers

Team name: TFRL

Date: 7/21/20

## System Test Scenarios:

### Sprint 1:

**User Stories:**

1. As a researcher, I want a pipeline to load and preprocess the text dataset (IMDb)
2. As a researcher, I want to use a fine-tuned BERT-like model so that I can use it as a supervisor of my GPT-2 generator during PPO training
3. As a researcher, I want to build a text classification pipeline so that I can use it to predict the label of a sentence

- Scenario 1:
  1. Using IMDb dataset.
  2. Run the test code "BERT Train and Evaluation Test" from  [test.ipynb](../testing/test.ipynb) 

**User Stories:**

1.  As a researcher, I want a pipeline to load and preprocess the text dataset (IMDb)
2.  As a researcher, I want to perform the language modeling task using the GPT-2 model so that I can use it for weak-signal controlled generation task
3.  As a researcher, I want to build a text generation pipeline so that I can measure generation quality

- Scenario 2:
  1. Using IMDb dataset.
  2. Run the test code "GPT-2 Train and Evaluation Test" from  [test.ipynb](../testing/test.ipynb) 

### Sprint 2:

**User Stories:**

1.  As a researcher, I want to set up the training loop of GPT-2 with PPO training so that it can do controlled generation (IMDb)
2.  As a researcher, I want to come up with multiple sets of possible hyperparameters so that I can test which set works the best (IMDb)
3.  As a researcher, I want to fine-tune the GPT-2 model using PPO training so that it can serve as the core data augmentation model (IMDb)

- Scenario 1:
  1. Using IMDb dataset.
  2. Run the test code  [run_gpt2_ppo_language_modeling.py](../src/run_gpt2_ppo_language_modeling.py) 

**User Story:**

1.  As a researcher, I want to perform the language modeling task using the GPT-2 model so that I can use it for weak-signal controlled generation task (Yelp polarity)

- Scenario 2:
  1. Using Yelp polarity dataset.
  2. Run the test code   [run_gpt2_language_modeling.py](../src/run_gpt2_language_modeling.py) 

**User Story:**

1.  As a researcher, I want to use a fine-tuned BERT-like model so that I can use it as a supervisor of my GPT-2 generator during PPO training (Yelp polarity)

- Scenario 3:
  1. Using Yelp polarity dataset.
  2. Run the test code   [run_bert_text_classification.py](../src/run_bert_text_classification.py) 

**User Story:**

1.  As a researcher, I want to set up the training loop of GPT-2 using PPO training so that I can fine-tune it with reward learning (Yelp polarity)

- Scenario 3:
  1. Using Yelp polarity dataset.
  2. Run the test code   [run_gpt2_ppo_language_modeling.py](../src/run_gpt2_ppo_language_modeling.py) 

### Sprint 3:

**User Story:**

1.  As a researcher, I want high-level abstraction so that I can focus on the data augmentation instead of the model or code
2.  As a researcher, I want a fully functional pipeline so that I don't need to assemble the parts manually
3.  As a researcher, I want to generate extra training data so that I can examine its quality

- Scenario 1:
  1.  [data_pipeline.ipynb](../demo/data_pipeline.ipynb) to test data preprocess module.
  2.  [classifier.ipynb](../demo/classifier.ipynb) to test classifier module.
  3.  [generator.ipynb](../demo/generator.ipynb) to test generator module.
  4. Run the test code "BERT Train and Evaluation Test" and "GPT-2 Train and Evaluation Test" from  [test.ipynb](../testing/test.ipynb) to test metrics module.

## Unit tests:

 [test.ipynb](../testing/test.ipynb) 