import datasets
from transformers import AutoTokenizer
import os


datasets.disable_caching()

tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-40b")
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(">>SUFFIX<<")
tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids(">>ABSTRACT<<")


data = datasets.load_dataset("mxy680/quotes")
data = datasets.concatenate_datasets([data["train"], data["validation"], data["test"]])

data = data.filter(lambda x: len(x["quote"].split()) >= 3 and len(x["quote"].split()) <= 100, num_proc=os.cpu_count())

data = data.map(
    lambda batch: {
        "prompt": [
            f">>ABSTRACT<<One wise man said: {quote}<|endoftext|>" for quote in batch["quote"]
        ]
    },
    batched=True,
    num_proc=os.cpu_count(),
)

data = data.map(
    lambda batch: tokenizer(batch["prompt"], max_length=64, truncation=True, padding=False),
    batched=True,
    remove_columns=data.column_names,
    num_proc=os.cpu_count(),
)



train_test = data.train_test_split(test_size=0.2, seed=42)
print(tokenizer)
print(tokenizer.decode(train_test['train'][0]['input_ids']))

train_test = train_test.with_format("torch")


train_test.save_to_disk("processed_dataset")
