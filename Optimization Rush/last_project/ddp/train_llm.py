import argparse
import os
import time
import logging
import numpy
import random

import torch
from torch.utils.data import DataLoader
import wandb
import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
)

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record


LOGGER = logging.getLogger(__name__)


@record
def main():
    parser = _get_parser()
    args = parser.parse_args()

    dist.init_process_group()

    rank = dist.get_rank()
    local_rank = rank % torch.cuda.device_count()
    world_size = dist.get_world_size()

    logging.basicConfig(
        format=f"[rank={rank}] [%(asctime)s] %(levelname)s:%(message)s",
        level=logging.INFO,
    )

    LOGGER.info(os.environ)
    LOGGER.info(args)
    LOGGER.info(f"local_rank={local_rank} rank={rank} world_size={world_size}")

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    
    config = AutoConfig.from_pretrained(args.model_name, use_cache=False)
    with device:
        model = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=torch.float32,
            attn_implementation="flash_attention_2" if args.FA else None,
        )
    if args.compile:
        torch._dynamo.config.optimize_ddp=False #for compile compatibility
        model = torch.compile(model)
    model = DistributedDataParallel(model, device_ids=[local_rank])

    LOGGER.info(f"{sum(p.numel() for p in model.parameters())} model parameters")

    train_data = torch.randint(
        low=0,
        high=128256,
        size=(args.num_samples, args.seq_length),
        dtype=torch.long,
    )

    def collate_fn(samples):
        batch = torch.stack(samples)
        return {"input_ids": batch, "labels": batch.clone()}

    LOGGER.info(f"{len(train_data)} training samples")

    g = torch.Generator()
    g.manual_seed(args.seed)
    dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        worker_init_fn=_seed_worker,
        generator=g,
        sampler=DistributedSampler(train_data, drop_last=True),
    )
    LOGGER.info(f"{len(dataloader)} batches per epoch")

    optimizer = torch.optim.AdamW(model.parameters(), fused=args.fused_adam)

    if rank == 0:
        wandb.init(
            project="optimization-rush",
            name=args.experiment_name,
            id=args.experiment_name,
            save_code=True,
            config={
                "args": vars(args),
                "world_size": world_size,
            },
        )

    global_step = 0
    log_steps_counter = 0

    model.train()
    batches = iter(dataloader)

    torch.cuda.synchronize(device=device)
    start_time = time.time()

    for _ in tqdm.tqdm(range(len(dataloader)), disable=rank > 0):
        batch = next(batches)
        batch = {k: v.to(device=device) for k, v in batch.items()}

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.amp):
            outputs = model(**batch)

        optimizer.zero_grad()
        outputs.loss.backward()

        optimizer.step()

        global_step += 1
        log_steps_counter += 1

        if global_step % (len(dataloader) // args.num_logs) == 0:
            torch.cuda.synchronize(device=device)
            end_time = time.time()
            total_time_per_log = end_time - start_time

            tok_per_log_steps = (
                args.batch_size * args.seq_length * log_steps_counter * world_size
            )

            info = {
                **get_mem_stats(device),
                "tokens_per_second": tok_per_log_steps / total_time_per_log,
            }

            LOGGER.info(info)
            if rank == 0:
                wandb.log(info, step=global_step * args.batch_size )

            torch.cuda.reset_peak_memory_stats(device)
            log_steps_counter = 0

            torch.cuda.synchronize(device=device)
            start_time = time.time()


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def get_mem_stats(device=None):
    mem = torch.cuda.memory_stats(device)
    props = torch.cuda.get_device_properties(device)
    return {
        "total_mem_in_gb": 1e-9 * props.total_memory,
        "curr_alloc_in_gb": 1e-9 * mem["allocated_bytes.all.current"],
        "peak_alloc_in_gb": 1e-9 * mem["allocated_bytes.all.peak"],
        "curr_resv_in_gb": 1e-9 * mem["reserved_bytes.all.current"],
        "peak_resv_in_gb": 1e-9 * mem["reserved_bytes.all.peak"],
    }


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", default=None, required=True)
    parser.add_argument("--model-name", default=None, required=True)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--FA", action="store_true")
    parser.add_argument("--fused-adam", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--seq-length", default=1024, type=int)
    parser.add_argument("--num-samples", default=1024, type=int)
    parser.add_argument("--num-logs", default=16, type=int)

    return parser


if __name__ == "__main__":
    main()

