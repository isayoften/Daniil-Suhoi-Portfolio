from tqdm import tqdm
import transformers
import torch

from accelerate import Accelerator
from accelerate.utils import set_seed

from peft import get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoModelForCausalLM


from dataset import get_dataloader, load_dataset
import config


def main():

    set_seed(config.seed)
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id, quantization_config=config.bnb_config, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16
    )
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, config.lora_config)
    model.print_trainable_parameters()
    model.config.use_cache = False

    optim = bnb.optim.AdamW8bit(model.parameters(), lr=config.lr)

    data = load_dataset("processed_dataset")
    train_dataloader = get_dataloader(
        dataset=data["train"], batch_size=config.batch_size, tokenizer=tokenizer
    )
    val_dataloader = get_dataloader(
        dataset=data["test"], batch_size=config.batch_size*2, tokenizer=tokenizer
    )
    scheduler = transformers.get_cosine_schedule_with_warmup(optim, 
                                                             num_warmup_steps=config.warmup_ratio*config.num_steps, 
                                                             num_training_steps=config.num_steps*config.scheduler_rate)

    model, optim, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optim, train_dataloader, val_dataloader, scheduler
    )

    if config.load_state:
        accelerator.load_state("last_state")

    model.train()
    train_loss = 0
    for i, batch in enumerate(tqdm(train_dataloader)):
        with accelerator.accumulate(model):
            optim.zero_grad()

            loss = model(**batch).loss
            accelerator.backward(loss)
            optim.step()
            scheduler.step()

            train_loss += loss.detach()

        if i != 0 and i % config.logging_and_saving_state_step == 0:
            print(f"Step_{i}; Loss: {train_loss/config.logging_and_saving_state_step}")
            train_loss = 0
            accelerator.save_state("last_state")

        if i == config.num_steps:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                "model",
                save_function=accelerator.save,
            )
            break

    model.eval()
    val_loss = 0
    for batch in tqdm(val_dataloader):
        with torch.inference_mode():
            loss = model(**batch).loss
        val_loss += loss
    print(f"Val_loss: {val_loss/len(val_dataloader)}")


if __name__ == "__main__":
    main()
