from transformers import MarianMTModel, MarianTokenizer
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


class Translate(object):
    """A translator that translate from source language (src) to target language (tgt)"""

    def __init__(self, src, tgt, top_p):
        self.src = src
        self.tgt = tgt
        self.top_p = top_p
        self.label = ">>" + self.tgt + "<< "
        self.model_name = "Helsinki-NLP/opus-mt-" + self.src + "-" + self.tgt
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        self.model = MarianMTModel.from_pretrained(self.model_name)

    def translate(self, sentences):
        # add tgt labels
        sentences = [self.label + s for s in sentences]
        # translate and generate
        translated = self.model.generate(
            **self.tokenizer.prepare_translation_batch(sentences), top_p=self.top_p
        )
        # decode the generated sentences
        tgt_text = [
            self.tokenizer.decode(t, skip_special_tokens=True) for t in translated
        ]
        return tgt_text
