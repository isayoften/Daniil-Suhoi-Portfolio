from datasets import load_dataset, disable_caching, DatasetDict
import os
from transformers import AutoTokenizer
disable_caching()


train_data, test_data = load_dataset(
    "smangrul/ultrachat-10k-chatml", split=["train", "test"]
) #small subset of the ultrachat

train_data = train_data.select_columns(["messages"])
test_data = test_data.select_columns(["messages"])


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct") #already have the correct EOS token

tokenizer.pad_token = "<|finetune_right_pad_id|>"
tokenizer.model_max_length = 2048


tokenized_train_data = train_data.map(
    lambda samples: {
        "input_ids": tokenizer.apply_chat_template(
            samples["messages"],
            add_generation_prompt=False,
            tokenize=True,
            truncation=True,
        )
    },
    batched=True,
    num_proc=os.cpu_count(),
    remove_columns=["messages"],
)
tokenized_eval_data = test_data.map(
    lambda samples: {
        "input_ids": tokenizer.apply_chat_template(
            samples["messages"],
            add_generation_prompt=False,
            tokenize=True,
            truncation=True,
        )
    },
    batched=True,
    num_proc=os.cpu_count(),
    remove_columns=["messages"],
)

combined_dataset = DatasetDict({
    'train': tokenized_train_data,
    'test': tokenized_eval_data
}).with_format("torch")

combined_dataset.save_to_disk('data')