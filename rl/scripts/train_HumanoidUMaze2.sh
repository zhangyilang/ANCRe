uv run train.py \
    --env_id "humanoid_u_maze" \
    --critic_depth 16 \
    --actor_depth 16 \

uv run train.py \
    --env_id "humanoid_u_maze" \
    --critic_depth 16 \
    --actor_depth 16 \
    --use_ancre \
    --ancre_softmax_temp 1e-1

# uv run train.py \
#     --env_id "humanoid_u_maze" \
#     --critic_depth 16 \
#     --actor_depth 16 \
#     --use_ancre \
#     --ancre_softmax_temp 1e-2

# uv run train.py \
#     --env_id "humanoid_u_maze" \
#     --critic_depth 16 \
#     --actor_depth 16 \
#     --use_ancre \
#     --ancre_softmax_temp 1e-3
