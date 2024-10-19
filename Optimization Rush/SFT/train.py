from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser, TrainingArguments, Trainer
from trl import DataCollatorForCompletionOnlyLM
import torch


def main():
    
    parser = HfArgumentParser(TrainingArguments)
    sft_config = parser.parse_args_into_dataclasses()[0]
    
    data = load_from_disk('data')

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    tokenizer.pad_token = "<|finetune_right_pad_id|>"
    tokenizer.model_max_length = 2048

    model_id = "meta-llama/Llama-3.2-3B"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        attn_implementation="flash_attention_2",
    )

    collator = DataCollatorForCompletionOnlyLM(
        response_template=[128006, 78191, 128007, 271], # "<|start_header_id|>assistant<|end_header_id|>\n\n"
        instruction_template=[128006, 882, 128007, 271], # "<|start_header_id|>user<|end_header_id|>\n\n"
        tokenizer=tokenizer,
        padding_free=True,
    )

    trainer = Trainer(
        model=model,
        args=sft_config,
        data_collator=collator,
        train_dataset=data['train'],
        eval_dataset=data['test'],
        tokenizer=tokenizer,
    )

    if sft_config.resume_from_checkpoint:
        print('hello')
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    trainer.save_model(sft_config.output_dir)


if __name__ == "__main__":
    main()