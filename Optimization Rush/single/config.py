import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig
import os


model_id = "meta-llama/Meta-Llama-3.1-8B"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

lora_config = LoraConfig(
    r=8, lora_dropout=0.05, task_type="CAUSAL_LM", target_modules="all-linear"
)

training_args = {
    "output_dir": "model",
    "per_device_train_batch_size": 64,
    "per_device_eval_batch_size": 128,
    "gradient_accumulation_steps": 1,
    "learning_rate": 3e-4,
    "weight_decay": 1e-3,
    "num_train_epochs": 1,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "logging_steps": 0.3,
    "eval_strategy": "steps",
    "log_level": "info",
    "save_strategy": "steps",
    "save_steps": 0.3,
    "save_total_limit": 1,
    "bf16": True,
    "dataloader_num_workers": os.cpu_count(),
    "optim": "paged_adamw_8bit",
    "group_by_length": True,
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
    "max_seq_length": 64,
}

resume_training = False
