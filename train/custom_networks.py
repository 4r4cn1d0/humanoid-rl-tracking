"""Custom network architectures for SAC with LayerNorm and SiLU activation."""

from __future__ import annotations

import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class LayerNormMLP(nn.Module):
    """MLP with LayerNorm after each hidden layer and SiLU activation."""

    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CustomActorCriticPolicy(ActorCriticPolicy):
    """
    Custom SAC policy with LayerNorm and SiLU activation.
    Network architecture: [512, 512, 256] for both actor and critic.
    """

    def __init__(self, *args, **kwargs):
        # Extract custom network architecture if provided
        net_arch = kwargs.pop("net_arch", None)
        if net_arch is None:
            net_arch = {"pi": [512, 512, 256], "qf": [512, 512, 256]}
        
        # Store net_arch for later use
        self._custom_net_arch = net_arch
        
        # Set activation function to SiLU
        kwargs["activation_fn"] = nn.SiLU
        
        super().__init__(*args, **kwargs)
        
        # Override the actor and critic networks with custom LayerNorm MLPs
        self._build_custom_networks(net_arch)

    def _build_mlp(self, input_dim: int, hidden_dims: list[int], output_dim: int) -> LayerNormMLP:
        """Build a custom MLP with LayerNorm and SiLU."""
        return LayerNormMLP(input_dim, hidden_dims, output_dim)

    def _build_custom_networks(self, net_arch: dict) -> None:
        """Build custom actor and critic networks with LayerNorm."""
        # Get input dimension from features extractor
        # For dict observations, SB3 uses a CombinedExtractor
        if hasattr(self.features_extractor, 'features_dim'):
            input_dim = self.features_extractor.features_dim
        else:
            # Fallback: compute from observation space
            if isinstance(self.observation_space, spaces.Dict):
                # Flatten dict observation
                input_dim = sum(space.shape[0] if len(space.shape) > 0 else 1 
                              for space in self.observation_space.spaces.values())
            else:
                input_dim = self.observation_space.shape[0]
        
        # Build actor network
        actor_hidden = net_arch.get("pi", [512, 512, 256])
        self.actor = self._build_mlp(input_dim, actor_hidden, self.action_space.shape[0])
        
        # Build critic networks (twin Q-networks)
        critic_hidden = net_arch.get("qf", [512, 512, 256])
        self.qf1 = self._build_mlp(input_dim + self.action_space.shape[0], critic_hidden, 1)
        self.qf2 = self._build_mlp(input_dim + self.action_space.shape[0], critic_hidden, 1)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Forward pass through actor network."""
        features = self.extract_features(obs)
        actions = self.actor(features)
        return actions

    def q_forward(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through critic networks."""
        features = self.extract_features(obs)
        q1_input = torch.cat([features, actions], dim=-1)
        q2_input = torch.cat([features, actions], dim=-1)
        return self.qf1(q1_input), self.qf2(q2_input)

    def _get_constructor_parameters(self) -> dict[str, any]:
        """Get constructor parameters for saving."""
        data = super()._get_constructor_parameters()
        data.update({
            "net_arch": self._custom_net_arch,
        })
        return data
