export TORCHELASTIC_ERROR_FILE=./error.json
export OMP_NUM_THREADS=8
torchrun --standalone \
    --nproc-per-node gpu \
    --redirects 3 \
    --log-dir ./logs \
    train_llm.py \
    --experiment-name llama_1B_ddp4gpu_bs4_1 \
    --model-name meta-llama/Llama-3.2-1B \
    --batch-size 4 \
    --amp \
    --FA \
    --fused-adam \
    --compile

