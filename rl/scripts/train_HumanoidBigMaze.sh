# uv run train.py \
#     --env_id "humanoid_big_maze" \
#     --critic_depth 64 \
#     --actor_depth 64 \

# uv run train.py \
#     --env_id "humanoid_big_maze" \
#     --critic_depth 64 \
#     --actor_depth 64 \
#     --use_ancre \
#     --ancre_softmax_temp 1e-2

uv run train.py \
    --env_id "humanoid_big_maze" \
    --critic_depth 64 \
    --actor_depth 64 \
    --use_ancre \
    --ancre_softmax_temp 1e0

uv run train.py \
    --env_id "humanoid_big_maze" \
    --critic_depth 64 \
    --actor_depth 64 \
    --use_ancre \
    --ancre_softmax_temp 1e1
