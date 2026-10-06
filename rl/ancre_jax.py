from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn


class AdaptiveNeuralConnectionReassign(nn.Module):
    num_layers: int = 0
    softmax_temp: float = 1e-2

    def setup(self):
        assert self.num_layers >= 0

        # Single parameter tensor instead of ParameterList
        # Shape: (num_layers, num_layers)
        # Only [:i+1] is used for layer i
        self.skip_params = self.param(
            "skip_params",
            nn.initializers.zeros,
            (self.num_layers, self.num_layers),
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        all_hidden_states: Optional[list[jnp.ndarray]],
        layer_idx: int,
    ) -> tuple[jnp.ndarray, Optional[list[jnp.ndarray]]]:
        """
        Args:
            hidden_states: current layer output
            all_hidden_states: list of previous hidden states (None before first layer)
            layer_idx: -1 before first layer, [0, num_layers-1] otherwise
        """

        # Before the first layer
        if layer_idx == -1:
            return hidden_states, [hidden_states]

        assert all_hidden_states is not None

        # Softmax skip coefficients
        raw_coeffs = self.skip_params[layer_idx, : layer_idx + 1]
        coeffs = jax.nn.softmax(raw_coeffs / self.softmax_temp, axis=0)

        # Stack previous hidden states: (layer_idx+1, ...)
        stacked = jnp.stack(all_hidden_states, axis=0)

        # Weighted sum over skip connections
        skip_update = jnp.tensordot(stacked, coeffs, axes=(0, 0))
        hidden_states = hidden_states + skip_update

        # Maintain history
        if layer_idx < self.num_layers - 1:
            all_hidden_states = all_hidden_states + [hidden_states]
        else:
            all_hidden_states = None  # after last layer

        return hidden_states, all_hidden_states
