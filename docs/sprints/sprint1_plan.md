# Sprint 1 Plan

Product name: TAPT

Team name: TFRL (Transformers with Reward Learning)

Revision number: 2

Revision date: 7/5/20

Sprint completion date: 7/6/20

## Goal

1. Be able to load text datasets

2. Be able to fine-tune BERT-like text classification models

3. Be able to fine-tune GPT-2 models

4. Be able to classify sentiments (BERT)

5. Be able to generate sentences similar to the training data (GPT-2)

## Tasks List

**User Story 1:** As a researcher, I want a pipeline to load and preprocess the IMDb dataset

| Task Index | Task                                                      | Task Time |
| :--------: | --------------------------------------------------------- | :-------: |
|     1      | Use HuggingFace NLP to load the IMDb dataset              |     1     |
|     2      | Split the training data into training and validation data |     1     |
|     3      | Tokenize training, validation, and test datasets          |     1     |
|     4      | Define input IDs and attention masks                      |     1     |
| **Total**  |                                                           |   **4**   |

**User Story 2:** As a researcher, I want to perform the language modeling task using the GPT-2 model so that I can use it for weak-signal controlled generation task.

| Task Index | Task                                                         | Task Time |
| :--------: | ------------------------------------------------------------ | :-------: |
|     1      | Prepare IMDb dataset in .txt format (each label followed by a sentence) |     1     |
|     2      | Define initial training arguments and trainer                |     1     |
|     3      | Test the limit of GPU memory consumption                     |     1     |
|     4      | Fine-tune GPT-2 model                                        |     2     |
|     5      | Evaluate the model on the validation set and tune hyperparameters |     6     |
|     6      | Report test set performance                                  |     1     |
|     6      | Save model                                                   |     1     |
| **Total**  |                                                              |  **13**   |

**User Story 3:** As a researcher, I want to use a fine-tuned BERT-like model so that I can use it as a supervisor of my GPT-2 generator during PPO training.

| Task Index | Task                       | Task Time |
| :--------: | -------------------------- | :-------: |
|     1      | Define initial training arguments and trainer |    1    |
|     2      | Test the limit of GPU memory consumption |     1     |
|     3      | Fine-tune BERT model |    2    |
| 4 | Evaluate the model on the validation set and tune hyperparameters |  6  |
| 5 | Report test set performance | 1 |
| 6 | Save model | 1 |
| **Totala** | | **12** |

**User Story 4:** As a researcher, I want to build a text classification pipeline so that I can use it to predict the label of a sentence.

| Task Index | Task                                            | Task Time |
| :--------: | ----------------------------------------------- | :-------: |
|     1      | Load fine-tuned model and its tokenizer         |     1     |
|     2      | Use HuggingFace pipeline to classify sentiments |     1     |
|     3      | Test the pipeline using sample sentences        |     2     |
| **Total**  |                                                 |   **4**   |

**User Story 5:** As a researcher, I want to build a text generation pipeline so that I can measure generation quality.

| Task Index | Task                                    | Task Time |
| :--------: | --------------------------------------- | :-------: |
|     1      | Load fine-tuned model and its tokenizer |     1     |
|     2      | Write generation function               |     2     |
|     3      | Define generation arguments             |     1     |
|     4      | Generate some sample sentences as test  |     2     |
| **Total**  |                                         |   **6**   |


## Team Roles

Chris Liu: Researcher, product owner

Jinfan Zhang: Developer, scrum master

Yiyun Zheng: Developer, team member

Tianyao Zheng: Developer, team member

## Initial Task Assignment

Chris Liu: user story 3, task 1

Yiyun Zheng: user story 2, task 1

Jinfan Zhang: user story 1, task 1

Tianyao Zheng: user story 1, task 1

## Initial Scrum Board

Scrum board link: https://trello.com/b/keCT4DuC

## Scrum Times

Monday 8 pm - 9 pm

Wednesday 8 pm - 9 pm

**With TA:**  Wednesday 3 pm - 4 pm

