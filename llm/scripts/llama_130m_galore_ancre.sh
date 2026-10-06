# LLaMA-130M, 1 A100, 1 Node
torchrun --standalone --nproc_per_node 1 torchrun_main.py \
    --model_config configs/llama_130m.json \
    --lr 5e-2 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer galore_adamw \
    --rank 256 \
    --use_ancre \
    --name GaLore+ANCRe \
    --disable_tqdm