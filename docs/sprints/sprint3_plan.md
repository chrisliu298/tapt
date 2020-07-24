# Sprint 3 Plan

Product name: TAPT

Team name: TFRL

Revision number: 2

Revision date: 

Sprint completion date:

## Goal

1. Be able to generated reviews given short prompts
2. Be able to evaluate generated text (IMDb and Yelp) by training a model using the augmented data
3. Be able to evaluate generated text (IMDb and Yelp) by human evaluation
4. Be able to write clean code and comments

## Tasks List

**User story 1:** As a researcher, I want to evaluate the augmented data by human examination so that I know the quality of the generated text.

| Task Index | Task                                                                                                | Task Time |
| :--------: | --------------------------------------------------------------------------------------------------- | --------- |
|     1      | Examine some portion of the augmented text data                                                     |     6     |
|     2      | Retrieve the low quality augmented data (those with opposite labels) using the trained BERT model   |     1     |
|     3      | Analyze what goes wrong with the augmented text with low quality                                    |     4     |
| **Total**  |                                                                                                     |     11    |

**User story 2:** As a researcher, I want to evaluate the augmented data by training a Transformer/RNN model so that I know the quality of the generated text.
| Task Index | Task                                                                           | Task Time |
| :--------: | ------------------------------------------------------------------------------ | --------- |
|     1      | Preprocess the augmented data (load it for training)                           |     1     |
|     2      | Train a BERT model as a classifier using both versions of the augmented text   |     2     |
|     3      | Evaluate the two BERT models' performance                                      |     1     |
| **Total**  |                                                                                |     4     |

**User story 3:** As a researcher, I want high-level abstraction so that I can focus on the data augmentation instead of the model or code.
| Task Index | Task                             | Task Time |
| :--------: | -------------------------------- | --------- |
|     1      | Refract GPT-2 training code      |    2      |
|     2      | Refract BERT training code       |    2      |
|     3      | Refract PPO-GPT-2 training code  |    3      |
|     4      | Refract data loader              |    2      |
| **Total**  |                                  |    9      |

**User story 4:** As a researcher, I want to see fully comment code so that I can refer to the source code when I need to.
| Task Index | Task                                         | Task Time |
| :--------: | -------------------------------------------- | --------- |
|     1      | Add full comments to GPT-2 training code     |    1      |
|     2      | Add full comments to BERT training code      |    1      |
|     3      | Add full comments to PPO-GPT-2 training code |    1      |
|     4      | Add full comments to data loader             |    1      |
| **Total**  |                                              |    4      |

**User sotry 5:** As a researcher, I want to see correctly formatted code so that it is easy to read.
| Task Index | Task                            | Task Time |
| :--------: | ------------------------------- | --------- |
|     1      | Use Black to format all codes   |     1     |
| **Total**  |                                 |     1     |

**User story 6:**  As a researcher, I want to generate extra training data so that I can examine its quality
| Task Index | Task                                                    | Task Time |
| :--------: | ------------------------------------------------------- | --------- |
|     1      | Generate 5000 training data using GPT-2                 |     16    |
|     2      | Generate 5000 training data using GPT-2 (PPO trained)   |     16    |
| **Total**  |                                                         |     32    |


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

