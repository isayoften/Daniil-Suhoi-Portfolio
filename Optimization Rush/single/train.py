from tqdm import tqdm
import transformers
import torch
from datasets import load_from_disk


from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)

from trl import SFTTrainer, SFTConfig

import config


def main():

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(
        "<|finetune_right_pad_id|>"
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        quantization_config=config.bnb_config,
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    data = load_from_disk("processed_dataset")

    training_args = SFTConfig(**config.training_args)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        peft_config=config.lora_config,
    )

    trainer.train(resume_from_checkpoint=config.resume_training)

    trainer.save_model()


if __name__ == "__main__":
    main()
