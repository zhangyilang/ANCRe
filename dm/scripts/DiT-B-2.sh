torchrun --standalone --nproc_per_node 2 train.py \
    --num-workers 2 \
    --image-size 256 \
    --model DiT-B/2 \
    --bf16 \
    --wandb-name FullPT