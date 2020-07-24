### Sprint 1

1.  As a researcher, I want a pipeline to load and preprocess the text dataset (IMDb)
2.  As a researcher, I want to perform the language modeling task using the GPT-2 model so that I can use it for weak-signal controlled generation task
3.  As a researcher, I want to use a fine-tuned BERT-like model so that I can use it as a supervisor of my GPT-2 generator during PPO training
4.  As a researcher, I want to build a text classification pipeline so that I can use it to predict the label of a sentence
5.  As a researcher, I want to build a text generation pipeline so that I can measure generation quality

### Sprint 2

1.  As a researcher, I want to set up the training loop of GPT-2 with PPO training so that it can do controlled generation (IMDb)
2.  As a researcher, I want to come up with multiple sets of possible hyperparameters so that I can test which set works the best (IMDb)
3.  As a researcher, I want to perform the language modeling task using the GPT-2 model so that I can use it for weak-signal controlled generation task (Yelp polarity)
4.  As a researcher, I want to use a fine-tuned BERT-like model so that I can use it as a supervisor of my GPT-2 generator during PPO training (Yelp polarity)
5.  As a researcher, I want to set up the training loop of GPT-2 using PPO training so that I can fine-tune it with reward learning (Yelp polarity)
6.  As a researcher, I want to fine-tune the GPT-2 model using PPO training so that it can serve as the core data augmentation model (IMDb)

### Sprint 3

1.  As a researcher, I want to evaluate the augmented data by human examination so that I know the quality of the generated text
2.  As a researcher, I want to evaluate the augmented data by training a Transformer/RNN model so that I know the quality of the generated text
3.  As a researcher, I want high-level abstraction so that I can focus on the data augmentation instead of the model or code
4.  As a researcher, I want to see fully comment code so that I can refer to the source code when I need to
5.  As a researcher, I want to see correctly formatted code so that it is easy to read
6.  As a researcher, I want a fully functional pipeline so that I don't need to assemble the parts manually
7.  As a researcher, I want to generate extra training data so that I can examine its quality

