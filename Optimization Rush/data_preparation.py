import datasets
from transformers import AutoTokenizer
import json
import os
import config
from collections import Counter
from tqdm import tqdm

datasets.disable_caching()

tokenizer = AutoTokenizer.from_pretrained(config.model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id

data = datasets.load_dataset("csv", data_files="dataset.zip")

# convert '['a', 'b']' to 'a, b'
data = data.map(
    lambda batch: {
        "NER": [", ".join(json.loads(x)).lower() for x in batch["NER"]],
        "directions": ["\n".join(json.loads(x)) for x in batch["directions"]],
    },
    batched=True,
    num_proc=os.cpu_count(),
)

ingr_count = Counter()
for ingr in tqdm(data["train"]["NER"]):
    ingr_count.update(ingr.split(", "))
    
# remove too short/long recipies, too short/long ingr list, and recipies with rare ingr (adjust numbers for your needs)
data = data.filter(
    lambda x: len(x["directions"].split()) >= 50
    and len(x["directions"].split()) <= 200
    and len(x["NER"].split(", ")) >= 3
    and len(x["NER"].split(", ")) <= 15
    and len(x["directions"].split("\n")) >= 2
    and all(ingr_count[ingredient] >= 100 for ingredient in x["NER"].split(", ")),
    num_proc=os.cpu_count(),
)

# make prompt
data = data.map(
    lambda batch: {
        "prompt": [
            f"The great chef shared a recipe for a very tasty dish. Ingredients: {ner}. Cooking instructions:\n{directions}"
            for ner, directions in zip(batch["NER"], batch["directions"])
        ]
    },
    batched=True,
    num_proc=os.cpu_count(),
)

# tokenize prompt
data = data.map(
    lambda batch: tokenizer(batch["prompt"], max_length=320, truncation=True),
    batched=True,
    remove_columns=data["train"].column_names,
    num_proc=os.cpu_count(),
)

train_test = data["train"].train_test_split(test_size=0.1, seed=42)

# sort dataset for efficient batching
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
