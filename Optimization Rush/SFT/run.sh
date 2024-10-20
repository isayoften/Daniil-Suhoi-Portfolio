accelerate launch train.py \
--output_dir checkpoints \
--overwrite_output_dir True \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 8 \
--gradient_accumulation_steps 1 \
--learning_rate 2e-5 \
--weight_decay 0.001 \
--max_grad_norm 1 \
--num_train_epochs 1 \
--lr_scheduler_type cosine \
--warmup_ratio 0.1 \
--optim adamw_torch_fused \
--use_liger_kernel True \
--dataloader_num_workers 16 \
--logging_steps 1 \
--save_steps 0.5 \
--eval_steps 0.25 \
--log_level info \
--eval_strategy steps \
--save_total_limit 1 \
--report_to wandb \
--seed 42 \

