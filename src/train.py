import torch

# import wandb
import time
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from random import choices
import matplotlib.pyplot as plt

from transformers import GPT2Tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from trl.gpt2 import GPT2HeadWithValueModel, respond_to_batch
from trl.ppo import PPOTrainer
from trl.core import build_bert_batch_from_txt


tqdm.pandas()

# Global training configuration
config = {
    "lm_name": "lvwerra/gpt2-imdb",
    "ref_lm_name": "lvwerra/gpt2-imdb",
    "cls_model_name": "lvwerra/bert-imdb",
    "tk_name": "gpt2",
    "steps": 51200,
    "batch_size": 128,
    "forward_batch_size": 16,
    "ppo_epochs": 4,
    "txt_in_len": 5,
    "txt_out_len": 50,
    "lr": 1.41e-5,
    "init_kl_coef": 0.2,
    "target": 6,
    "horizon": 10000,
    "gamma": 1,
    "lam": 0.95,
    "cliprange": 0.2,
    "cliprange_value": 0.2,
    "vf_coef": 0.1,
    "seed": 1,
}

# Set random seed
np.random.seed(config["seed"])

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pos_logit_to_reward(logit, task):
    """
    Take the positive sentiment logit and scale it for the task.
        task [negative]: reward = -logit
        task [neutral]: reward = -2*abs(logit)+4
        task [positive]: reward = logit
    """
    for i in range(len(logit)):
        if task[i] == "[negative]":
            logit[i] = -logit[i]
        elif task[i] == "[neutral]":
            logit[i] = -2 * torch.abs(logit[i]) + 4
        elif task[i] == "[positive]":
            pass
        else:
            raise ValueError("task has to be in [0, 1, 2]!")
    return logit


if __name__ == "__main__":

    # wandb.login()
    # wandb.init(name='long-response', project='gpt2-control', config=config)

    # Load dataset
    df = pd.read_csv("nlp_imdb_train.tsv", delimiter="\t")

    # Keep only the first 1000 characters
    df["text"] = df["text"].apply(lambda x: x[:1000])

    # Load tokenizers and models
    bert_model = AutoModelForSequenceClassification.from_pretrained(
        config["cls_model_name"]
    )
    bert_tokenizer = AutoTokenizer.from_pretrained(config["cls_model_name"])

    gpt2_model = GPT2HeadWithValueModel.from_pretrained(config["lm_name"])
    gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained(config["ref_lm_name"])
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(config["tk_name"])

    # Move models to GPU
    gpt2_model.to(device)
    bert_model.to(device)
    gpt2_model_ref.to(device)

    # Tokenize the first n BPE tokens in the sentence
    df["tokens"] = df["text"].progress_apply(
        lambda x: gpt2_tokenizer.encode(" " + x, return_tensors="pt").to(device)[
            0, : config["txt_in_len"]
        ]
    )

    # Decode the sequence to get the original first n BPE words
    df["query"] = df["tokens"].progress_apply(lambda x: gpt2_tokenizer.decode(x))

    control_str = ["[negative]", "[neutral]", "[positive]"]
    control_tokens = dict(
        (s, gpt2_tokenizer.encode(s, return_tensors="pt").squeeze().to(device))
        for s in control_str
    )

    ppo_trainer = PPOTrainer(gpt2_model, gpt2_model_ref, **config)
    fbs = config["forward_batch_size"]
    # wandb.watch(gpt2_model, log="all")

    ppo_trainer = PPOTrainer(gpt2_model, gpt2_model_ref, **config)
    fbs = config["forward_batch_size"]

    for epoch in tqdm(range(int(np.ceil(config["steps"] / config["batch_size"])))):
        torch.cuda.empty_cache()
        logs = dict()
        game_data = dict()
        timing = dict()
        t0 = time.time()

        # get a batch from the dataset and annotate tasks
        df_batch = df.sample(config["batch_size"])
        task_list = choices(control_str, k=config["batch_size"])
        task_tensors = torch.stack([control_tokens[t] for t in task_list])
        query_list = df_batch["query"].tolist()
        game_data["query"] = [t + q for t, q in zip(task_list, query_list)]

        query_tensors = torch.stack(df_batch["tokens"].tolist())
        query_tensors = torch.cat((task_tensors, query_tensors), axis=1)

        # get response from gpt2
        t = time.time()
        response_tensors = []
        for i in range(int(config["batch_size"] / fbs)):
            response = respond_to_batch(
                gpt2_model,
                query_tensors[i * fbs : (i + 1) * fbs],
                txt_len=config["txt_out_len"],
            )
            response_tensors.append(response)
        response_tensors = torch.cat(response_tensors)
        game_data["response"] = [
            gpt2_tokenizer.decode(response_tensors[i, :])
            for i in range(config["batch_size"])
        ]
        timing["time/get_response"] = time.time() - t

        # tokenize text for sentiment analysis
        t = time.time()
        texts = [q + r for q, r in zip(query_list, game_data["response"])]
        sentiment_inputs, attention_masks = build_bert_batch_from_txt(
            texts, bert_tokenizer, device
        )
        timing["time/build_input_sentiment"] = time.time() - t

        # get sentiment score
        t = time.time()
        pos_logits = []
        for i in range(int(config["batch_size"] / fbs)):
            res = bert_model.forward(
                sentiment_inputs[i * fbs : (i + 1) * fbs],
                attention_masks[i * fbs : (i + 1) * fbs],
            )[0][:, 1].detach()
            pos_logits.append(res)
        rewards = pos_logit_to_reward(torch.cat(pos_logits), task_list)
        timing["time/get_sentiment_preds"] = time.time() - t

        # Run PPO training
        t = time.time()
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        timing["time/optimization"] = time.time() - t

        # Log everything
        timing["time/epoch"] = time.time() - t0
        table_rows = [
            list(r)
            for r in zip(
                game_data["query"], game_data["response"], rewards.cpu().tolist()
            )
        ]
        #     logs.update({'game_log':wandb.Table(
        #         columns=['query', 'response', 'reward'],
        #         rows=table_rows)})
        logs.update(timing)
        logs.update(stats)
        logs["env/reward_mean"] = torch.mean(rewards).cpu().numpy()
        logs["env/reward_std"] = torch.std(rewards).cpu().numpy()
        logs["env/reward_dist"] = rewards.cpu().numpy()
        for control_s in control_str:
            key = "env/reward_" + control_s.strip("[]")
            logs[key] = np.mean(
                [
                    r
                    for r, t in zip(logs["env/reward_dist"], task_list)
                    if t == control_s
                ]
            )
        # wandb.log(logs)

    # os.makedirs('gpt2-imdb')
    # gpt2_model.save_pretrained('gpt2-imdb')
    # gpt2_tokenizer.save_pretrained('gpt2-imdb')
