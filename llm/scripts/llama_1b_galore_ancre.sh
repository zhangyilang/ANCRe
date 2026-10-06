# LLaMA-1B, 8 A100, 1 Node
torchrun --standalone --nproc_per_node 2 torchrun_main.py \
    --model_config configs/llama_1b.json \
    --lr 5e-3 \
    --batch_size 16 \
    --total_batch_size 512 \
    --num_training_steps 100000 \
    --warmup_steps 10000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer galore_adamw \
    --rank 512 \
    --use_ancre \
    --name GaLore+ANCRe \
    --disable_tqdm