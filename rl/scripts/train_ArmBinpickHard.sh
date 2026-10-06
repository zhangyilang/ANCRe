# Arm Binpick Hard: ResNet-16 and ResNet-64, without and with ANCRe. Run from rl/.
declare -A TEMP=([16]=1e-2 [64]=1e-1)

for DEPTH in 16 64; do
    uv run train.py --env_id arm_binpick_hard --critic_depth $DEPTH --actor_depth $DEPTH
    uv run train.py --env_id arm_binpick_hard --critic_depth $DEPTH --actor_depth $DEPTH \
        --use_ancre --ancre_softmax_temp ${TEMP[$DEPTH]}
done
