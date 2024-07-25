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
        device_placement=False,
    )

    device = accelerator.device

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id, quantization_config=config.bnb_config, device_map='auto'
    )
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
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
        dataset=data["validation"], batch_size=config.batch_size*2, tokenizer=tokenizer
    )
    # scheduler = transformers.get_cosine_schedule_with_warmup(optim, warmup_steps, num_epochs*scheduler_rate*len(train_dataloader))

    model, optim, train_dataloader, val_dataloader = accelerator.prepare(
        model, optim, train_dataloader, val_dataloader
    )

    if config.load_state:
        accelerator.load_state("last_state")

    model.train()
    train_loss = 0
    for i, batch in enumerate(tqdm(train_dataloader)):
        with accelerator.accumulate(model):
            optim.zero_grad()

            batch = {k: v.to(device) for k, v in batch.items()}

            loss = model(**batch).loss
            accelerator.backward(loss)
            optim.step()

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
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.inference_mode():
            loss = model(**batch).loss
        val_loss += loss
    print(f"Val_loss: {val_loss/len(val_dataloader)}")


if __name__ == "__main__":
    main()
