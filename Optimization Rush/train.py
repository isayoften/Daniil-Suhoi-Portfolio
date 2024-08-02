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
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>")

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        quantization_config=config.bnb_config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.pad_token_id = tokenizer.pad_token_id    

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=config.gradient_checkpointing
    )
    model = get_peft_model(model, config.lora_config)
    model.print_trainable_parameters()
    model.config.use_cache = False

    optim = bnb.optim.PagedAdamW8bit(model.parameters(), lr=config.lr)

    data = load_dataset("processed_dataset")
    train_dataloader = get_dataloader(
        dataset=data["train"], batch_size=config.batch_size, tokenizer=tokenizer
    )
    val_dataloader = get_dataloader(
        dataset=data["test"], batch_size=config.batch_size, tokenizer=tokenizer
    )
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=config.num_epochs
        * len(train_dataloader)
        * config.warmup_ratio,
        num_training_steps=config.num_epochs
        * len(train_dataloader)
        * config.scheduler_rate,
    )

    model, optim, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optim, train_dataloader, val_dataloader, scheduler
    )

    if config.load_state:
        accelerator.load_state("last_state")

    best_val = 1e10

    for epoch in range(config.num_epochs):

        model.train()
        train_loss = 0
        for batch in tqdm(train_dataloader):
            with accelerator.accumulate(model):
                optim.zero_grad()

                loss = model(**batch).loss
                accelerator.backward(loss)
                optim.step()
                scheduler.step()

                train_loss += loss.detach()

        train_loss /= len(train_dataloader)

        model.eval()
        val_loss = 0
        for batch in tqdm(val_dataloader):
            with torch.no_grad():
                loss = model(**batch).loss
                val_loss += loss.detach()

        val_loss /= len(val_dataloader)

        print(f"Epoch  {epoch}; Train_loss = {train_loss}, Val_loss = {val_loss}")
        
        accelerator.save_state("last_state")
        
        if val_loss < best_val:
            best_val = val_loss
            accelerator.unwrap_model(model).save_pretrained(
                "model",
                save_function=accelerator.save,
            )


if __name__ == "__main__":
    main()
