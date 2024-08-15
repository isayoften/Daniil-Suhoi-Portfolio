import torch
from datasets import load_from_disk
from peft.utils.other import fsdp_auto_wrap_policy


from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)

from trl import SFTTrainer

import config


def main():

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(">>SUFFIX<<")
    tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids(">>ABSTRACT<<")
   
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=torch.bfloat16,
        quantization_config=config.bnb_config,
        use_cache=False,
        attn_implementation="flash_attention_2",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id


    data = load_from_disk("../processed_dataset")

    trainer = SFTTrainer(
        model=model,
        args=config.training_args,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        peft_config=config.lora_config,
    )

    fsdp_plugin = trainer.accelerator.state.fsdp_plugin
    fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)


    trainer.train(resume_from_checkpoint=config.resume_training)

    trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()

    


if __name__ == "__main__":
    main()
