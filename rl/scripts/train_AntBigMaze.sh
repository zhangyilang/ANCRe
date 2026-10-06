uv run train.py \
    --env_id "ant_big_maze" \
    --eval_env_id "ant_big_maze_eval" \
    --critic_depth 64 \
    --actor_depth 64 \

uv run train.py \
    --env_id "ant_big_maze" \
    --eval_env_id "ant_big_maze_eval" \
    --critic_depth 64 \
    --actor_depth 64 \
    --use_ancre \
    --ancre_softmax_temp 1e-2
