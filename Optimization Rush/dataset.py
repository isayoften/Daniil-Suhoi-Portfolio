import torch
import datasets
import random
from transformers import DataCollatorForLanguageModeling, AutoTokenizer
import os
import config

# tokenizer = AutoTokenizer.from_pretrained(config.model_id)
# tokenizer.pad_token_id = tokenizer.eos_token_id
# data = datasets.load_from_disk("processed_dataset")


# for out in tokenizer.batch_decode(data['train'].shuffle(42).select(range(10))['input_ids']):
#     print()
#     print(out)


def load_dataset(path):
    dataset = datasets.load_from_disk(path)
    print("Dataset loaded")
    return dataset

# efficient batching with minimal padding. Split sorted data into batches and then shuffle batches order
class EfficientBatchSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size

        self.sorted_batches = [
            list(range(i, min(i + batch_size, len(data_source))))
            for i in range(0, len(data_source), batch_size)
        ]

        random.shuffle(self.sorted_batches)

    def __iter__(self):
        return iter([idx for batch in self.sorted_batches for idx in batch])

    def __len__(self):
        return len(self.data_source)


def get_dataloader(dataset, batch_size, tokenizer):

    sampler = EfficientBatchSampler(dataset, batch_size)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=os.cpu_count(),
        pin_memory=True,
        collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    return dataloader
