import datasets
from transformers import AutoTokenizer
import os
import config

datasets.disable_caching()

tokenizer = AutoTokenizer.from_pretrained(config.model_id, add_eos_token=True)
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>")

data = datasets.load_dataset("Abirate/english_quotes")

data = data.filter(lambda x: len(x["quote"].split()) >= 3, num_proc=os.cpu_count())

data = data.map(
    lambda batch: {
        "prompt": [f"One wise man said: {quote}<|end_of_text|>" for quote in batch["quote"]]
    },
    batched=True,
    num_proc=os.cpu_count(),
)

data = data.map(
    lambda batch: tokenizer(batch["prompt"], max_length=128, truncation=True),
    batched=True,
    remove_columns=data["train"].column_names,
    num_proc=os.cpu_count(),
)

train_test = data["train"].train_test_split(test_size=0.2, seed=42)

train_test = train_test.map(
    lambda batch: {"lens": [len(x) for x in batch["input_ids"]]},
    batched=True,
    num_proc=os.cpu_count(),
)

train_test["train"] = train_test["train"].sort("lens")
train_test["test"] = train_test["test"].sort("lens")
train_test = train_test.remove_columns(["lens"])
train_test = train_test.with_format("torch")


train_test.save_to_disk("processed_dataset")
