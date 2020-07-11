# Sprint 2 Plan

Product name: TAPT

Team name: TFRL (Transformers with Reward Learning)

Revision number: 1

Revision date: 7/7/20

Sprint completion date: 7/14/20

## Goal

1. Be able to train GPT-2 with PPO
2. Be able to generate sentences with higher quality by using optimized GPT-2
3. Be able to fine-tune BERT-like text classification models with Yelp polarity
4. Be able to fine-tune GPT-2 models with Yelp polarity
5. Be able to classify sentiments (BERT)
6. Be able to generate sentences similar to the training data (GPT-2)

## Tasks List

**User Story 1:** As a researcher, I want to set up the training loop of GPT-2 with PPO training so that it can do controlled generation (IMDb)

| Task Index | Task                                        | Task Time |
| :--------: | ------------------------------------------- | :-------: |
|     1      | Define PPO training arguments               |     1     |
|     2      | Define logit to reward function             |     1     |
|     3      | Preprocess IMDb dataset (load and truncate) |     1     |
|     4      | Load GPT-2 model                            |     1     |
|     5      | Load BERT model                             |     1     |
| **Total**  |                                             |     5     |

**User Story 2:** As a researcher, I want to come up with multiple sets of possible hyperparameters so that I can test which set works the best (IMDb)

| Task Index | Task                                 | Task Time |
| :--------: | ------------------------------------ | :-------: |
|     1      | Training loop log                    |     1     |
|     2      | Hyperparameter tuning (PPO training) |     1     |
|     3      | Generate sentences samples           |     1     |
| **Total**  |                                      |     3     |

**User Story 3:** As a researcher, I want to fine-tune the GPT-2 model using PPO training so that it can serve as the core data augmentation model (IMDb)

| Task Index | Task                                                         | Task Time |
| :--------: | ------------------------------------------------------------ | :-------: |
|     1      | PPO training loop: GPT-2 generating short sentences          |     2     |
|     2      | PPO training loop: BERT assigning reward to the generated sentences |     2     |
|     3      | PPO training loop: Calculate sentiment reward                |     2     |
|     4      | PPO training loop: GPT-2 optimization                        |    14     |
| **Total**  |                                                              |  **20**   |

**User Story 4:** As a researcher, I want to perform the language modeling task using the GPT-2 model so that I can use it for weak-signal controlled generation task (Yelp polarity)

| Task Index | Task                                                         | Task Time |
| :--------: | ------------------------------------------------------------ | :-------: |
|     1      | Prepare Yelp polarity dataset in .txt format (each label followed by a sentence) |     1     |
|     4      | Fine-tune GPT-2 model                                        |     2     |
|     5      | Evaluate the model on the validation set and tune hyperparameters |     6     |
|     6      | Report test set performance                                  |     1     |
|     6      | Save model                                                   |     1     |
| **Total**  |                                                              |  **11**   |

**User Story 5:** As a researcher, I want to use a fine-tuned BERT-like model so that I can use it as a supervisor of my GPT-2 generator during PPO training (Yelp polarity)

| Task Index | Task                                                         | Task Time |
| :--------: | ------------------------------------------------------------ | :-------: |
|     1      | Fine-tune BERT model 1                                       |     2     |
|     2      | Fine-tune BERT model 0                                       |     2     |
|     3      | Evaluate the model on the validation set and tune hyperparameters (Modify epochs) |     6     |
|     4      | Evaluate the model on the validation set and tune hyperparameters (Modify warm up steps) |     6     |
|     5      | Evaluate the model on the validation set and tune hyperparameters (Modify data set size) |    12     |
|     6      | Report test set performance                                  |     1     |
|     7      | Save model                                                   |     1     |
| **Total**  |                                                              |  **30**   |

**User Story 6:** As a researcher, I want to set up the training loop of GPT-2 using PPO training so that I can fine-tune it with reward learning (Yelp polarity)

| Task Index | Task                                                         | Task Time |
| :--------: | ------------------------------------------------------------ | :-------: |
|     1      | PPO training loop: GPT-2 generating short sentences          |     2     |
|     2      | PPO training loop: BERT assigning reward to the generated sentences |     2     |
|     3      | PPO training loop: Calculate sentiment reward                |     2     |
|     4      | PPO training loop: GPT-2 optimization                        |    14     |
| **Total**  |                                                              |  **20**   |



## Team Roles

Chris Liu: Researcher, product owner

Yiyun Zheng: Developer, scrum master

Jinfan Zhang: Developer, team member

Tianyao Zheng: Developer, team member

## Initial Task Assignment

Chris Liu: user story 1, task 1

Yiyun Zheng: user story 5, task 2

Jinfan Zhang: user story 5, task 1

Tianyao Zheng: user story 4, task 1

## Initial Scrum Board

Scrum board link: https://chrisliu298.atlassian.net/secure/RapidBoard.jspa?rapidView=1

## Scrum Times

Monday 8 pm - 9 pm

Wednesday 8 pm - 9 pm

**With TA:**  Wednesday 3 pm - 4 pm

