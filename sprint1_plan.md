# Sprint 1 Plan

Product name: TAPT

Team name: TFRL (Transformers with Reward Learning)

Revision number: 1

Revision date: 6/29/20

Sprint completion date: 7/6/20

## Goal

1. Be able to preprocess .txt and tabular dataset

2. Be able to fine-tune BERT-like text classification model

3. Be able to fine-tune GPT-2 model

4. Be able to classify sentiments (BERT)

5. Be able to generate positive/negative sentences (GPT-2)

## Tasks List

For all the tasks related to datasets, everyone will be assigned one dataset. We are going to use:

1.  IMDb
2.  Spam
3.  Toxic comments
4.  Yelp polarity

**User Story 1:** As a user, I want a pipeline to preprocess .txt files so that I can have the data in the correct format to feed into the tokenizer.

| Task Index | Task                                                         | Task Time |
| :--------: | ------------------------------------------------------------ | :-------: |
|     1      | Implement Python file handler                                |    0.1    |
|     2      | Slice each sentence (string) so that it only keeps the first 1000 characters |    0.1    |
|     3      | If the string length is less than 1000, do nothing           |    0.1    |
|     4      | Output the final dataset as a Pandas dataframe with columns "text" and "label" |    0.2    |
| **Total**  |                                                              |  **0.5**  |

**User Story 2:** As a user, I want a pipeline to preprocess tabular data so that I can have the data in the correct format to feed into the tokenizer.

| Task Index | Task                                                         | Task Time |
| :--------: | ------------------------------------------------------------ | :-------: |
|     1      | Implement Pandas csv/tsv reader                              |    0.1    |
|     2      | Slice each sentence (string) so that it only keeps the first 1000 characters |    0.1    |
|     3      | If the string length is less than 1000, do nothing           |    0.1    |
|     4      | Output the final dataset as a Pandas dataframe with columns "text" and "label" |    0.2    |
| **Total**  |                                                              |  **0.5**  |

**User Story 3:** As a researcher, I want to perform the text classification task using BERT-like models so that I can train a state-of-the-art text classifier.

| Task Index | Task                       | Task Time |
| :--------: | -------------------------- | :-------: |
|     1      | Implement BERT tokenizer   |    0.5    |
|     2      | Write BERT training loop   |     1     |
|     3      | Write BERT evaluation loop |     1     |
| **Total**  |                            |  **2.5**  |

**User Story 4:** As a researcher, I want to perform the language modeling task using the GPT-2 model so that I can have a customizable GPT-2 text generator of text in any genre.

Task 1: Implement GPT-2 tokenizer.
Task 2: Use HuggingFace Trainer (instead of writing the training loop).
Task 3: Fine-tune GPT-2.

| Task Index | Task                                                         | Task Time |
| :--------: | ------------------------------------------------------------ | :-------: |
|     1      | Implement GPT-2 tokenizer                                    |    0.5    |
|     2      | Use HuggingFace Trainer (instead of writing the training loop) |     2     |
|     3      | Fine-tune GPT-2                                              |     3     |
| **Total**  |                                                              |  **5.5**  |

**User Story 5:** As a researcher, I want to perform the language modeling task using the GPT-2 model so that I can use it for a weak-signal controlled generation task.

| Task Index | Task                                                         | Task Time |
| :--------: | ------------------------------------------------------------ | :-------: |
|     1      | Implement GPT-2 tokenizer                                    |    0.5    |
|     2      | Use HuggingFace Trainer (instead of writing the training loop) |     2     |
|     3      | Fine-tune GPT-2 with weak label signal                       |     4     |
| **Total**  |                                                              |  **6.5**  |


**User Story 6:** As a researcher, I want to use a fine-tuned BERT-like model so that I can use it as a supervisor of my GPT-2 generator during PPO training.

Task 1: Write BERT model loader so that we can load it as a text classifier.

| Task Index | Task                      | Task Time |
| :--------: | ------------------------- | :-------: |
|     1      | Implement GPT-2 tokenizer |    0.5    |
| **Total**  |                           |  **0.5**  |

**User Story 7:** As a researcher, I want to build a text classification pipeline so that I can use it to evaluate test accuracy.

| Task Index | Task                         | Task Time |
| :--------: | ---------------------------- | :-------: |
|     1      | Write BERT evaluation script |     2     |
|     2      | Evaluate BERT                |     1     |
| **Total**  |                              |   **3**   |

**User story 8:** As a researcher, I want to build a text generation pipeline so that I can measure generation quality.

| Task Index | Task                               | Task Time |
| :--------: | ---------------------------------- | :-------: |
|     1      | Write GPT-2 generation script      |     1     |
|     2      | Use GPT-2 to generate some samples |     2     |
| **Total**  |                                    |   **3**   |


## Team Roles

Chris Liu - Researcher, product owner

Yiyun Zheng - Developer, scrum master

Jinfan Zhang - Developer

Tianyao Zheng - Developer

## Initial Task Assignment

Chris Liu - User Story 1，IMDb

Yiyun Zheng - User Story 1，Spam

Jinfan Zhang - User Story 1，Toxic

Tianyao Zheng - User Story 1，Yelp polarity

## Initial Scrum Board

Scrum board link: https://trello.com/b/keCT4DuC

## Scrum Times

Monday 8 pm - 9 pm

Wednesday 8 pm - 9 pm

**With TA:** Friday or Wednesday 3 pm - 4 pm

