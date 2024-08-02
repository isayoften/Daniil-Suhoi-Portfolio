import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig


model_id = "meta-llama/Meta-Llama-3.1-8B"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

lora_config = LoraConfig(
    r=8,
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    target_modules='all-linear' #highly reccomends using 'all-linear' in the paper
)

batch_size = 16
lr = 3e-4
num_epochs = 5
load_state = False

warmup_ratio = 0.1
scheduler_rate = 1.5

seed = 42
mixed_precision = "bf16" #A100 can run bf16
gradient_accumulation_steps = 1
gradient_checkpointing = True
