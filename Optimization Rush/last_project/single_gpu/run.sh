python train_llm.py \
    --experiment-name llama_1B_amp_fusedadam_FA_compile_bs16_gc_seqlen4096 \
    --model-name meta-llama/Llama-3.2-1B \
    --num-samples 256 \
    --batch-size 16 \
    --amp \
    --fused-adam \
    --FA \
    --compile \
    --gc \
    --seq-length 4096

