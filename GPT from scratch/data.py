from datasets import load_dataset, disable_caching
import os
from itertools import chain

from transformers import GPT2TokenizerFast

disable_caching()

tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")

dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    num_proc=os.cpu_count(),
    split="train",
)
dataset = dataset.select_columns(["text"])

max_seq_length = 1024 + 1 # +1 for shifted targets
def group_texts(examples):

    examples = tokenizer(
        [sample + "<|endoftext|>" for sample in examples["text"]],
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )["input_ids"]
    # Concatenate all texts.
    concatenated_examples = list(chain(*examples))
    total_length = len(concatenated_examples)
    # We drop the small remainder, and if the total_length < max_seq_length  we exclude this batch and return an empty dict.
    total_length = (total_length // max_seq_length) * max_seq_length
    # Split by chunks of max_len.
    result = {
        "input_ids": [
            concatenated_examples[i : i + max_seq_length]
            for i in range(0, total_length, max_seq_length)
        ]
    }
    return result


dataset = dataset.map(
    group_texts,
    batched=True,
    num_proc=os.cpu_count(),
    remove_columns=dataset.column_names,
    batch_size=5000
)

dataset = dataset.train_test_split(test_size=0.025, shuffle=True, seed=42)

dataset = dataset.with_format("torch")
dataset.save_to_disk("prepared_data", num_proc=os.cpu_count())
