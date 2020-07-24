# Sprint 3 Plan

Product name: TAPT

Team name: TFRL

Revision number: 2

Revision date: 

Sprint completion date:

## Goal

1. Be able to train GPT-2 with PPO
2. Be able to generate sentences with higher quality by using optimized GPT-2
3. Be able to fine-tune BERT-like text classification models with Yelp polarity
4. Be able to fine-tune GPT-2 models with Yelp polarity
5. Be able to classify sentiments (BERT)
6. Be able to generate sentences similar to the training data (GPT-2)

## Tasks List

**User story 1:** As a researcher, I want to evaluate the augmented data by human examination so that I know the quality of the generated text.

| Task Index | Task                                                                                                | Task Time |
| :--------: | --------------------------------------------------------------------------------------------------- | --------- |
|     1      | Examine some portion of the augmented text data                                                     |           |
|     2      | Retrieve the low quality augmented data (those with opposite labels) using the trained BERT model   |           |
|     3      | Analyze what goes wrong with the augmented text with low quality                                    |           |
| **Total**  |                                                                                                     |           |

**User story 2:** As a researcher, I want to evaluate the augmented data by training a Transformer/RNN model so that I know the quality of the generated text.
| Task Index | Task                                                                           | Task Time |
| :--------: | ------------------------------------------------------------------------------ | --------- |
|     1      | Preprocess the augmented data (load it for training)                           |           |
|     2      | Train a BERT model as a classifier using both versions of the augmented text   |           |
|     3      | Evaluate the two BERT models' performance                                      |           |
| **Total**  |                                                                                |           |

**User story 3:** As a researcher, I want high-level abstraction so that I can focus on the data augmentation instead of the model or code.
| Task Index | Task                        | Task Time |
| :--------: | --------------------------- | --------- |
|     1      | Refract GPT-2 trainer       |           |
|     2      | Refract BERT trainer        |           |
|     3      | Refract PPO-GPT-2 trainer   |           |
|     4      | Refract data loader         |           |
| **Total**  |                             |           |

**User story 4:** As a researcher, I want to see fully comment code so that I can refer to the source code when I need to.
| Task Index | Task                                   | Task Time |
| :--------: | -------------------------------------- | --------- |
|     1      | Add full comments to GPT-2 trainer     |           |
|     2      | Add full comments to BERT trainer      |           |
|     3      | Add full comments to PPO-GPT-2 trainer |           |
|     4      | Add full comments to data loader       |           |
| **Total**  |                                        |           |

**User sotry 5:** As a researcher, I want to see correctly formatted code so that it is easy to read.
| Task Index | Task                            | Task Time |
| :--------: | ------------------------------- | --------- |
|     1      | Use Black to format all codes   |           |
| **Total**  |                                 |           |

**User story 6:** As a researcher, I want a fully functional pipeline so that I don't need to assemble the parts manually.
| Task Index | Task                                                  | Task Time |
| :--------: | ----------------------------------------------------- | --------- |
|     1      | Write example code to demonstrate the full pipeline   |           |
| **Total**  |                                                       |           |

**User story 7:**  As a researcher, I want to generate extra training data so that I can examine its quality
| Task Index | Task                                                    | Task Time |
| :--------: | ------------------------------------------------------- | --------- |
|     1      | Generate 5000 training data using GPT-2                 |           |
|     2      | Generate 5000 training data using GPT-2 (PPO trained)   |           |
| **Total**  |                                                         |           |


## Team Roles

Chris Liu: Researcher, product owner

Tianyao Zheng: Developer, scrum master

Jinfan Zhang: Developer, team member

Yiyun Zheng: Developer, team member

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

