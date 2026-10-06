# ANCRe for Deep Reinforcement Learning

This directory contains the reinforcement learning experiments of ANCRe (Section 5.3 of the paper): goal-conditioned contrastive RL (CRL) with deep ResNet actors and critics on `humanoid`, `ant_big_maze`, `arm_push_hard`, and `arm_binpick_hard`.

The code is built on [scaling-crl](https://github.com/wang-kevin3290/scaling-crl) (Wang et al., 2025), which in turn builds on [JaxGCRL](https://github.com/MichalBortkiewicz/JaxGCRL). Both are released under the Apache License 2.0; see [LICENSE](LICENSE). Our modifications add the JAX implementation of ANCRe ([ancre_jax.py](ancre_jax.py)) and its integration into the residual networks in [train.py](train.py).

# Installation

```sh
uv sync
```
Then just fix the two Brax issues described below, and you'll be all set.


## Fixing two bugs in brax 0.10.1
1. There is a minor bug in brax's contact.py file. To fix it, first locate the brax contact.py file in your virtual environment: 
```
find .venv -name contact.py
```
Then open the file and replace it with the following code:
```python
from typing import Optional
from brax import math
from brax.base import Contact
from brax.base import System
from brax.base import Transform
import jax
from jax import numpy as jp
from mujoco import mjx

def get(sys: System, x: Transform) -> Optional[Contact]:
    """Calculates contacts.
    Args:
        sys: system defining the kinematic tree and other properties
        x: link transforms in world frame
    Returns:
        Contact pytree
    """
    #NOTE: THIS WAS MODIFIED SINCE AFTER MUJOCO 3.1.5, mjx.ncon IS NOT AVAILABLE
    # ncon = mjx.ncon(sys)
    # if not ncon:
    #   return None
    data = mjx.make_data(sys)
    if data.ncon == 0:
        return None
    @jax.vmap
    def local_to_global(pos1, quat1, pos2, quat2):
        pos = pos1 + math.rotate(pos2, quat1)
        mat = math.quat_to_3x3(math.quat_mul(quat1, quat2))
        return pos, mat
    x = x.concatenate(Transform.zero((1,)))
    xpos = x.pos[sys.geom_bodyid - 1]
    xquat = x.rot[sys.geom_bodyid - 1]
    geom_xpos, geom_xmat = local_to_global(
        xpos, xquat, sys.geom_pos, sys.geom_quat
    )
    # pytype: disable=wrong-arg-types
    d = data.replace(geom_xpos=geom_xpos, geom_xmat=geom_xmat)
    d = mjx.collision(sys, d)
    # pytype: enable=wrong-arg-types
    c = d.contact
    elasticity = (sys.elasticity[c.geom1] + sys.elasticity[c.geom2]) * 0.5
    body1 = jp.array(sys.geom_bodyid)[c.geom1] - 1
    body2 = jp.array(sys.geom_bodyid)[c.geom2] - 1
    link_idx = (body1, body2)
    return Contact(elasticity=elasticity, link_idx=link_idx, **c.__dict__)
```
2. There is also a minor bug in brax's json.py file. To fix it, first locate the brax json.py file in your virtual environment:
```
find .venv -name json.py | grep "/brax/io/json.py"
```
Then open the file and change the if statement in line 159 to:  
```python
if (rgba == jp.array([0.5, 0.5, 0.5, 1.0])).all():
```

# Running experiments

A GPU is required; Humanoid-based environments with deep networks may need up to 80GB of GPU memory. ANCRe is enabled with `--use_ancre`, and `--ancre_softmax_temp` sets the softmax temperature. For example, a 64-layer ResNet on Humanoid:

```sh
# Baseline: cascaded residual connections
uv run train.py --env_id "humanoid" --critic_depth 64 --actor_depth 64

# ANCRe
uv run train.py --env_id "humanoid" --critic_depth 64 --actor_depth 64 --use_ancre --ancre_softmax_temp 1e-2
```

The scripts used for the paper are in [scripts/](scripts/). Results are logged to Weights & Biases; logging options (`track`, `wandb_project_name`, ...) are defined in `Args` in [train.py](train.py).

# Citation

If you use this code, please cite ANCRe together with the work it builds on:

```bibtex
@inproceedings{zhang2026ancre,
  title     = {{ANCRe}: Adaptive Neural Connection Reassignment for Efficient Depth Scaling},
  author    = {Yilang Zhang and Bingcong Li and Niao He and Georgios B. Giannakis},
  booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems},
  year      = {2026},
  url       = {https://arxiv.org/abs/2602.09009}
}

@inproceedings{wang2025,
  title     = {1000 Layer Networks for Self-Supervised {RL}: Scaling Depth Can Enable New Goal-Reaching Capabilities},
  author    = {Kevin Wang and Ishaan Javali and Micha{\l} Bortkiewicz and Tomasz Trzcinski and Benjamin Eysenbach},
  booktitle = {The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year      = {2025},
  url       = {https://openreview.net/forum?id=s0JVsx3bx1}
}
```
