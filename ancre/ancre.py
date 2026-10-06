from dataclasses import dataclass

import torch
import torch.nn as nn

from .utils import HadamardLifting


PARAMETRIZATIONS = {"softmax", "hadamard", "constant"}
ACTIVATION_MAPPINGS = {"softmax": lambda: nn.Softmax(dim=0),
                       "hadamard": HadamardLifting,
                       "constant": lambda: nn.Softmax(dim=0)}
# "constant": frozen zero logits -> softmax yields uniform coefficients 1/n, independent of softmax_temp
INITIALIZATION_MAPPINGS = {"softmax": nn.init.zeros_,
                           "hadamard": nn.init.ones_,
                           "constant": lambda x: nn.init.zeros_(x).requires_grad_(False)}


@dataclass
class ANCReConfig:
    num_layers: int = 0
    parameterization: str = "Softmax"
    softmax_temp: float = 1e-2


class AdaptiveNeuralConnectionReassign(nn.Module):
    def __init__(self, config: ANCReConfig):
        super().__init__()
        self.config = config

        assert self.config.num_layers > 0, "num_layers must be greater than 0"
        assert self.config.parameterization.lower() in PARAMETRIZATIONS, (
            f"parameterization must be one of {PARAMETRIZATIONS}"
        )
        self.activation = ACTIVATION_MAPPINGS[self.config.parameterization.lower()]()
        self.register_buffer('softmax_temp', torch.tensor(self.config.softmax_temp))

        self.shortcut_params = nn.ParameterList()
        for i in range(self.config.num_layers):
            self.shortcut_params.append(nn.Parameter(torch.empty(i + 1)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.shortcut_params[0]).requires_grad_(False)
        for param in self.shortcut_params[1:]:
            INITIALIZATION_MAPPINGS[self.config.parameterization.lower()](param)

    def forward(self, hidden_states: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        # Before the first layer
        if layer_idx == -1:
            self.all_hidden_states = [hidden_states, ]
            return hidden_states

        shortcut_coeffs = self.activation(self.shortcut_params[layer_idx] * (1 / self.softmax_temp))
        # Memory-frugal weighted sum over history. Equivalent to
        #   torch.tensordot(torch.stack(self.all_hidden_states), shortcut_coeffs, dims=([0], [0]))
        # but avoids retaining the stacked (K, ...) tensor -> ~0 extra activation memory vs vanilla.
        update = shortcut_coeffs[0] * self.all_hidden_states[0]
        for coeff, past in zip(shortcut_coeffs[1:], self.all_hidden_states[1:]):
            update = update + coeff * past
        hidden_states = hidden_states + update

        if layer_idx < self.config.num_layers - 1:
            self.all_hidden_states.append(hidden_states)
        else:
            # After the last layer
            self.all_hidden_states = None
        
        return hidden_states


class ANCReOutgoingNorm(nn.Module):
    def __init__(self, config: ANCReConfig):
        super().__init__()
        self.config = config

        assert self.config.num_layers > 0, "num_layers must be greater than 0"
        assert self.config.parameterization.lower() in PARAMETRIZATIONS, (
            f"parameterization must be one of {PARAMETRIZATIONS}"
        )
        self.activation = ACTIVATION_MAPPINGS[self.config.parameterization.lower()]()
        self.register_buffer('softmax_temp', torch.tensor(self.config.softmax_temp))

        self.shortcut_params = nn.ParameterList()
        for i in range(self.config.num_layers):
            self.shortcut_params.append(nn.Parameter(torch.empty(self.config.num_layers - i)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.shortcut_params[-1]).requires_grad_(False)
        for param in self.shortcut_params[:-1]:
            INITIALIZATION_MAPPINGS[self.config.parameterization.lower()](param)

    def forward(self, hidden_states: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        # Before the first layer
        if layer_idx == -1:
            self.all_hidden_states = [hidden_states, ]
            return hidden_states

        for i in range(layer_idx + 1):
            shortcut_coeff = self.activation(self.shortcut_params[i] * (1 / self.softmax_temp))[layer_idx - i]
            hidden_states = hidden_states + self.all_hidden_states[i]  * shortcut_coeff

        if layer_idx < self.config.num_layers - 1:
            self.all_hidden_states.append(hidden_states)
        else:
            # After the last layer
            self.all_hidden_states = None
        
        return hidden_states


class NaiveLearnableResidualConnections(nn.Module):
    def __init__(self, config: ANCReConfig):
        super().__init__()
        self.config = config
        assert self.config.num_layers > 0, "num_layers must be greater than 0"

        self.shortcut_params = nn.Parameter(torch.empty(self.config.num_layers))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.ones_(self.shortcut_params)

    def forward(self, hidden_states: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        if layer_idx > 0:
            hidden_states = hidden_states + self.last_hidden_states * self.shortcut_params[layer_idx]
        self.last_hidden_states = hidden_states
        
        return hidden_states
