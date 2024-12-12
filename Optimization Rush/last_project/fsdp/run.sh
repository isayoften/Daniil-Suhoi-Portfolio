export TORCHELASTIC_ERROR_FILE=./error.json
export OMP_NUM_THREADS=8
torchrun --standalone \
    --nproc-per-node gpu \
    --redirects 3 \
    --log-dir ./logs \
    train_llm.py \
    --experiment-name llama_8B_fsdp_gc \
    --model-name meta-llama/Llama-3.1-8B \
    --batch-size 1 \
    --FA \
    --fused-adam \
    --gc

