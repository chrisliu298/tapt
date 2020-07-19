from transformers import pipeline


class Classifier:
    """A sentiment classifier.

    Attribute:
            model: The bert model object.
            tokenizer: The tokenizer object.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.ppl = pipeline(
            "sentiment-analysis", model=self.model, tokenizer=self.tokenizer
        )

    def classify(self, text):
        """Returns the sentiment of a sentence {0: negative, 1: positive}."""
        return self.ppl(text)
