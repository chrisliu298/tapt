# Third party
from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer


class Classifier:
    """A sentiment classifier.

    Attribute:
        model_name_or_path: The name or path of a BERT model.
        tokenizer_name_or_path: The name or path of a BERT tokenizer.

        The model/tokenizer names can be found here:
        https://huggingface.co/transformers/pretrained_models.html
    """

    def __init__(self, model_name_or_path, tokenizer_name_or_path):
        self.model_name_or_path = model_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name_or_path, use_fast=True
        )
        self.ppl = pipeline(
            "sentiment-analysis", model=self.model, tokenizer=self.tokenizer
        )

    def classify(self, text):
        """Returns the sentiment of a sentence {0: negative, 1: positive}."""
        return self.ppl(text)
