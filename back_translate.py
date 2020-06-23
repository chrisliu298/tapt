from transformers import MarianMTModel, MarianTokenizer
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


class Translate(object):
    """A translator that translate from source language (src) to target 
    language (tgt)"""

    def __init__(self, src, tgt, top_p=0.95):
        """Constructs a translator (from src to tgt).

        Args:
            src: The source language.
            tgt: The target language.
            top_p: The cumulative probability of parameter highest probability 
                vocabulary tokens to keep for nucleus sampling. Must be between 
                0 and 1. Default to 1.
        """
        self.src = src
        self.tgt = tgt
        self.top_p = top_p
        self.label = ">>" + self.tgt + "<< "
        self.model_name = "Helsinki-NLP/opus-mt-" + self.src + "-" + self.tgt
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        self.model = MarianMTModel.from_pretrained(self.model_name)

    def translate(self, sentences):
        """Translate a list of sentences from a source language to a target language."""

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


class BackTranslate(object):
    """A back-translator that translate from source language (src) to target
    language (tgt), and then does the reverse"""

    def __init__(self, src, tgt, top_p=0.5):
        """Constructs a back-translator (from src to tgt, and then from tgt to src).

        Args:
            src: The source language.
            tgt: The target language.
            top_p: The cumulative probability of parameter highest probability 
                vocabulary tokens to keep for nucleus sampling. Must be between 
                0 and 1. Default to 1.
        """
        self.src = src
        self.tgt = tgt
        self.top_p = top_p
        self.translator = Translate(self.src, self.tgt, top_p=self.top_p)
        self.back_translator = Translate(self.tgt, self.src, top_p=self.top_p)

    def back_translate(self, sentences):
        """Translate a list of sentences from a source language to a target language,
        and then does the reverse."""

        tgt_sentences = self.translator.translate(sentences)
        return self.back_translator.translate(tgt_sentences)
