# Release Plan
Product name: TAPT

Team name: TFRL

Release name: v0.2

Release date: n/a

Revision number: 2

Revision date: 7/5/20

## High-Level Goals

1. Be able to load text datasets
2. Be able to preprocess (tokenize) text datasets
3. Be able to fine-tune a BERT-like model (e.g., BERT, DistilBERT, RoBERTa, DistilRoBERTa, etc.)
4. Be able to fine-tune a GPT-2 model of any size (e.g., DistilGPT2, small, medium, large, xl)
5. Be able to predict the label of a sentence
6. Be able to generate a continuation given a label and a prompt
7. Be able to fine-tune GPT-2 with PPO training
8. Be able to evaluate the augmented (generated) text


## User Stories for Release

### Sprint 1

| User Stories                                                 | Story Points |
| ------------------------------------------------------------ | :----------: |
| As a researcher, I want a pipeline to load and preprocess the text dataset (IMDb) |      2       |
| As a researcher, I want to perform the language modeling task using the GPT-2 model so that I can use it for weak-signal controlled generation task. |      3       |
| As a researcher, I want to use a fine-tuned BERT-like model so that I can use it as a supervisor of my GPT-2 generator during PPO training. |      3       |
| As a researcher, I want to tune the hyperparameters of the BERT model so that I can have the best performing model. |      6       |
| As a researcher, I want to tune the hyperparameters of the GPT-2 model so that I can have the best performing model. |      6       |
| As a researcher, I want to build a text classification pipeline so that I can use it to predict the label of a sentence. |      2       |
| As a researcher, I want to build a text generation pipeline so that I can measure generation quality. |      2       |



### Sprint 2

| User Stories                                                 | Story Points |
| ------------------------------------------------------------ | :----------: |
| As a researcher, I want to set up the training loop of GPT-2 using PPO training so that I can fine-tune it with reward learning. |      6       |
| As a researcher, I want to come up with multiple sets of possible hyperparameters so that I can test which set works the best. |      2       |
| As a researcher, I want to repeat the experiment using the other dataset (Yelp polarity) so that I can see if I can get similar results. |      8       |



### Sprint 3

| User Stories                                                 | Story Points |
| ------------------------------------------------------------ | :----------: |
| As a researcher, I want an RNN model so that I can evaluate the quality of the augmented text. |      4       |
| As a researcher, I want a Transformer model so that I can evaluate the quality of the augmented text. |      4       |
| As a researcher, I want a metric so that the augmented text can be compared with other types of augmentations. |      3       |
| As a user, I want everything as a class (Python object) so that I can work with them without spending too much time adapting to the tool. |      4       |




## Product Backlog

This section is empty because everything discussed during the release meeting will be added to the release.

## Project Presentation

[Project presentation link](https://docs.google.com/presentation/d/1db1pVHyLvHRHqmT50MzUQGD9WSmV5X51vY9Aw_k7BDE/edit?usp=sharing)