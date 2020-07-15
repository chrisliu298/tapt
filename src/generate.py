import collections
from dict2obj import Dict2Obj
import numpy as np
import torch
import pandas as pd
import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoConfig, set_seed


class Generate:
    def __init__(
        self,
        model_type,
        model_name_or_path,
        num_return_sequences=1,
        length=512,
        repetition_penalty=1.0,
        temperature=1.0,
        k=0,
        p=0.9,
        no_cuda=False,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        seed=42,
    ):
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.num_return_sequences = num_return_sequences
        self.length = length
        self.repetition_penalty = repetition_penalty
        self.temperature = temperature
        self.k = k
        self.p = p
        self.no_cuda = no_cuda
        self.device = device
        self.seed = seed
        self.MAX_LENGTH = int(10000)
        self.start_token = "<|startoftext|>"
        self.sep_token = "<|sep|>"
        self.stop_token = "<|endoftext|>"

    def set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)

    def adjust_length_to_model(self, length, max_sequence_length):
        if length < 0 and max_sequence_length > 0:
            length = max_sequence_length
        elif 0 < max_sequence_length < length:
            length = max_sequence_length
        elif length < 0:
            length = self.MAX_LENGTH
        return length

    def generate(self, tokenizer, model, prompt):
        set_seed(self.seed)
        self.length = self.adjust_length_to_model(
            self.length, max_sequence_length=model.config.max_position_embeddings
        )
        prompt_text = self.start_token + " " + prompt
        encoded_prompt = tokenizer.encode(
            prompt_text, add_special_tokens=False, return_tensors="pt"
        )
        encoded_prompt = encoded_prompt.to(self.device)

        input_ids = None if encoded_prompt.size()[-1] == 0 else encoded_prompt
        output_sequences = model.generate(
            input_ids=input_ids,
            max_length=self.length + len(encoded_prompt[0]),
            temperature=self.temperature,
            top_k=self.k,
            top_p=self.p,
            repetition_penalty=self.repetition_penalty,
            do_sample=True,
            num_return_sequences=1,
        )
        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_sequences = []
        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            generated_sequence = generated_sequence.tolist()
            text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
            text = text[: text.find(self.stop_token) if self.stop_token else None]
            total_sequence = (
                prompt_text
                + text[
                    len(
                        tokenizer.decode(
                            encoded_prompt[0], clean_up_tokenization_spaces=True
                        )
                    ) :
                ]
            )
            generated_sequences.append(total_sequence)
        return generated_sequences
