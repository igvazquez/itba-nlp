from transformers import (
    AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling,
    Trainer, TrainingArguments
)
from transformers.trainer_callback import PrinterCallback
from transformers.utils.notebook import NotebookProgressCallback
from datasets import load_dataset, DatasetDict

dataset = load_dataset("text", data_files="dataset/trump_tweets.txt")


dataset = DatasetDict(
    train=dataset["train"].shuffle(seed=33),
    val=dataset["train"].shuffle(seed=33),
    test=dataset["train"].shuffle(seed=33),
)

model_checkpoint = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

tokens_ = tokenizer.tokenize(dataset['train']["text"][0])