from peft import LoraConfig
import os
from trl import SFTConfig
import torch
from transformers import BitsAndBytesConfig


model_id = "tiiuae/falcon-40b"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_storage=torch.bfloat16,
)


lora_config = LoraConfig(
    r=8,
    lora_dropout=0.05, 
    task_type="CAUSAL_LM",
    target_modules="all-linear"
)

training_args = SFTConfig(
    output_dir = "model",
    per_device_train_batch_size = 16,
    per_device_eval_batch_size = 32,
    gradient_accumulation_steps = 1,
    learning_rate = 3e-4,
    weight_decay = 1e-2,
    max_steps=400,
    lr_scheduler_type = 'constant',
    logging_steps = 50,
    eval_strategy = "steps",
    eval_steps=400,
    log_level = "info",
    save_strategy = "no",
    save_total_limit = 1,
    bf16 = True,
    dataloader_num_workers = os.cpu_count(),
    optim = "adamw_torch",
    group_by_length = True,
    gradient_checkpointing = True,
    gradient_checkpointing_kwargs = {"use_reentrant": True},
    max_seq_length = 64,

)

resume_training = False
