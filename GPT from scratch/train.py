import torch
from datasets import load_from_disk
import os
from tqdm import tqdm
import wandb
from gpt2_model import GPT, GPTConfig
from accelerate import Accelerator
from accelerate.utils import set_seed

from transformers import get_cosine_schedule_with_warmup


def main(args):
    set_seed(42)

    accelerator = Accelerator(
        gradient_accumulation_steps=args["grad_accum_steps"],
        mixed_precision=args["mixed_precision"],
        log_with="wandb",
    )
    accelerator.init_trackers("gpt2 from scratch", config=args) # for logging

    dataset = load_from_disk("prepared_data")

    def data_collator(batch):
        x = torch.stack([sample["input_ids"][:-1] for sample in batch]) # shifted inputs and targets for causal LM
        y = torch.stack([sample["input_ids"][1:].clone().detach() for sample in batch])
        return (x, y)

    train_dataloader = torch.utils.data.DataLoader(
        dataset["train"],
        batch_size=args["per_device_micro_batch_size"],
        num_workers=os.cpu_count(),
        pin_memory=True,
        shuffle=True,
        collate_fn=data_collator,
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset["test"],
        batch_size=args["per_device_micro_batch_size"] * 2,
        num_workers=os.cpu_count(),
        pin_memory=True,
        shuffle=False,
        collate_fn=data_collator,
    )

    model = GPT(GPTConfig(vocab_size=50304))  # 50257 - orig vocab size, but 50304 = 2^7 * 393 so matmuls will be faster
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args["lr"],
        betas=args["betas"],
        weight_decay=args["weight_decay"],
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args["max_steps"] * args["warmup_ratio"],
        num_training_steps=args["max_steps"] * args["scheduler_ratio"],
    )

    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )

    if args['load_state']:
        accelerator.wait_for_everyone()
        accelerator.load_state('states')


    train_dataloader_iter = iter(train_dataloader) # because i want to use dataloader without for loop

    for step in range(args["max_steps"]):
        if (step + 1)*10 > len(train_dataloader):
            train_dataloader_iter = iter(train_dataloader) # reset dataloader at the end

        # validating
        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for x, y in tqdm(val_dataloader):
                    with accelerator.autocast():
                        _, loss = model(x, y)
                    val_loss += loss.detach()
                val_loss = val_loss / len(val_dataloader)
                val_loss = accelerator.reduce(val_loss, reduction='mean') #averaging losses across GPUs
                accelerator.print(f"val loss {val_loss.item():.4f}")
                accelerator.log({"val_loss": val_loss.item()}, step=step)
            

        # saving
        if (step+1) % 100 == 0:
            accelerator.wait_for_everyone()
            accelerator.save_state("states", safe_serialization=False)
            accelerator.wait_for_everyone()
            accelerator.save_model(accelerator.unwrap_model(model), "model", safe_serialization=False)

        # training
        model.train()
        optimizer.zero_grad()
        loss_accum = 0
        for accum_step in range(args["grad_accum_steps"]): #gradient accumulation for maintaining original gpt2 batch size
            with accelerator.accumulate(model): # accelerator will handle proper accumulation and synchronization
                x, y = next(train_dataloader_iter)
                with accelerator.autocast():
                    _, loss = model(x, y)
                loss_accum += loss.detach()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    norm = accelerator.clip_grad_norm_(model.parameters(), args["max_grad_norm"])
                optimizer.step()
                scheduler.step()
        loss_accum = loss_accum / args["grad_accum_steps"]
        loss_accum = accelerator.reduce(loss_accum, reduction='mean')      
        print(f"step {step} | train_loss {loss_accum.item():.4f} | norm {norm:.4f} | lr {optimizer.param_groups[0]['lr']}")
        accelerator.log(
            {
                "train_loss": loss_accum,
                "grad_norm": norm,
                "lr": optimizer.param_groups[0]["lr"],
            },
            step=step,
        )

    accelerator.end_training()


if __name__ == "__main__":

    args = dict(
        total_batch_size=524288,  # orig batch size in tokens
        per_device_micro_batch_size=8,
        sequence_length=1024,
        num_of_gpus=2,
        mixed_precision="bf16",
        max_grad_norm=1,
        lr=6e-4,
        warmup_ratio=0.05,
        scheduler_ratio=1.1, 
        betas=(0.9, 0.95),
        weight_decay=0.1,
        max_steps=10000, #actual optimizer steps
        load_state = False
    )

    assert args["total_batch_size"] % (args['per_device_micro_batch_size'] * args['sequence_length'] * args['num_of_gpus']) == 0, "make sure total_batch_size is divisible by B * seq_len * num_gpu"
    # we want to use original gpt2 batch size regardless of the memory dependent micro batch size
    args['grad_accum_steps'] = args['total_batch_size'] // (args['per_device_micro_batch_size'] * args['sequence_length'] * args['num_of_gpus'])
    main(args)
