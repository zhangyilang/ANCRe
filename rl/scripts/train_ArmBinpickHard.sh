# uv run train.py \
#     --env_id "arm_binpick_hard" \
#     --critic_depth 64 \
#     --actor_depth 64 \

uv run train.py \
    --env_id "arm_binpick_hard" \
    --critic_depth 64 \
    --actor_depth 64 \
    --use_ancre \
    --ancre_softmax_temp 1e-1
