# CSE115A Introduction to Software Engineering Group Project

This repository is for CSE115A (summer 2020) group project.


## TAPT: Text Augmentation Using Pre-Trained Transformers

The more data we have, and the large the model is, the better our performance. For many of today's natural language processing (NLP) downstream tasks (e.g., text classification, question answering, etc.), the dataset plays an essential role in whether the model is successful, and one of the qualities of the dataset is its size. While annotating the dataset is luxury, data augmentation becomes the key here. Unlike image data augmentation, which we can achieve easily with different transformation techniques or color inversion, it is hard to augment text because of one of the nature of human languages - high complexity.

![](images/tapt.png)

In this project, we will build a text data augmentation pipeline based on Transformer models. Transformer is one of the most significant advances in recent years. A Transformer model uses the attention mechanism, supports parallelization, and obliterates recurrence (Vaswani et al., 2017). With parallelization, the Transformer model runs faster on modern GPUs, and is more shallow and thus easier to optimize without the nature of being recurrent. In our case, we use GPT-2, a Transformer with 1.5B parameters trained on approximately 40 billion tokens of text, obtained by scraping all outbound links from Reddit with at least three upvotes (Radford et al., 2019), to generate new supervised training data based on the labels.

Besides conditional generation, we will also incorporate proximal policy optimization (PPO) (Schulman et al., 2017), a policy gradient method for reinforcement learning. PPO allows us to control the generation of GPT-2. For example, in the case of sentiment analysis, GPT-2 can generate a positive movie review conditioned on a positive label. Specifically, we will use BERT (Devlin et al., 2018) as a supervisor during the fine-tuning of GPT-2, which evaluates GPT-2's generation and rewards it if a generated sentence is "very" positive (or vice versa).

This project is inspired by Ziegler et al. (2019) and Kumar et al. (2020).



## Presentations

[Initial presentation slides link](https://docs.google.com/presentation/d/1db1pVHyLvHRHqmT50MzUQGD9WSmV5X51vY9Aw_k7BDE/edit?usp=sharing)

[Final presentation slides link]()


### References

Brown, Tom B., et al. "Language models are few-shot learners." arXiv preprint arXiv:2005.14165 (2020).

Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." arXiv preprint arXiv:1810.04805 (2018).

Kumar, Varun, Ashutosh Choudhary, and Eunah Cho. "Data augmentation using pre-trained transformer models." arXiv preprint arXiv:2003.02245 (2020).

Radford, Alec, et al. "Language models are unsupervised multitask learners." OpenAI Blog 1.8 (2019): 9.

Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).

Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems. 2017.

Ziegler, Daniel M., et al. "Fine-tuning language models from human preferences." arXiv preprint arXiv:1909.08593 (2019).



## Members

- Chris Liu

- Jinfan Zhang

- Yiyun Zheng

- Tianyao Zheng
