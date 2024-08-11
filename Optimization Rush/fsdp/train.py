from tqdm import tqdm
import transformers
import torch


from accelerate import Accelerator
from accelerate.utils import set_seed

from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM


from dataset import get_dataloader, load_dataset
import config

from peft.utils.other import fsdp_auto_wrap_policy


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

    # model = prepare_model_for_kbit_training(
    #     model, 
    #     use_gradient_checkpointing=config.gradient_checkpointing, 
    #     gradient_checkpointing_kwargs = {"use_reentrant": True}
    # )

    model.gradient_checkpointing_enable({"use_reentrant": False})

    model = get_peft_model(model, config.lora_config)
    model.print_trainable_parameters()
    model.config.use_cache = False

    accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)
    
    optim = torch.optim.AdamW(model.parameters(), lr=config.lr)

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

    
    for epoch in range(config.num_epochs):

        model.train()
        train_loss = 0
        for i, batch in enumerate(tqdm(train_dataloader, disable=not accelerator.is_local_main_process)):
            with accelerator.accumulate(model):
                optim.zero_grad()

                loss = model(**batch).loss
                accelerator.backward(loss)
                optim.step()
                scheduler.step()

                train_loss += loss.detach()
                

            if i!=0 and i % config.log_step == 0:

                train_loss /= config.log_step
                train_loss = accelerator.reduce(train_loss, reduction='mean')

                model.eval()
                val_loss = 0
                for batch in tqdm(val_dataloader, disable=not accelerator.is_local_main_process):
                    with torch.no_grad():
                        loss = model(**batch).loss
                        val_loss += loss.detach()
                        
                val_loss /= len(val_dataloader)
                val_loss = accelerator.reduce(val_loss, reduction='mean')

                accelerator.print(f"Train_loss = {train_loss}, Val_loss = {val_loss}")
                  
                train_loss = 0
                model.train()
    
    accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    accelerator.wait_for_everyone()
    accelerator.unwrap_model(model).save_pretrained(
        "model",
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
    


if __name__ == "__main__":
    main()
