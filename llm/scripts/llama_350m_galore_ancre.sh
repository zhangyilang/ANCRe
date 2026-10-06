# LLaMA-350M, 4 A100, 1 Node
torchrun --standalone --nproc_per_node 4 torchrun_main.py \
    --model_config configs/llama_350m.json \
    --lr 5e-2 \
    --batch_size 64 \
    --total_batch_size 512 \
    --num_training_steps 60000 \
    --warmup_steps 6000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer galore_adamw \
    --rank 256 \
    --use_ancre \
    --name GaLore+ANCRe \
    --disable_tqdm