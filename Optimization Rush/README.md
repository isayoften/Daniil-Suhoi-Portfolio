# Optimization Rush
Моя статья на HF: https://huggingface.co/blog/Isayoften/optimization-rush

check prev. projects in "old_projects"

curr_project in progress...
nice plots and results soon....

Цель - побенчить прирост скорости разных сетапов и методов оптимизации. Вот примерный план:
1. 1B llama single node вообще без оптимизаций
2. +mixed precision
3. +optimized kernels: FA, torch.compile, fused adamw
6. +большой батч (compute intensity)
7. +большой батч привел к большим активациям - activation checkpointing
9. рассказать про другие оптимизации, которые не были использованы: seq packing, quantization, accumulation, 
10. DDP 2 4 8
11. хотим модель побольше, но она не помещается? - FSDP/model parallelism






