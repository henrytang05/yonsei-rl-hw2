import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cas4160.infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(parameters, learning_rate)

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        # DONE: implement get_action
        obs_t = ptu.from_numpy(obs).unsqueeze(0)

        dist = self.forward(obs_t)
        action_t = dist.sample()

        action = ptu.to_numpy(action_t.squeeze(0))

        return action

    def forward(self, obs: torch.Tensor) -> distributions.Distribution:
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            # DONE: define the forward pass for a policy with a discrete action space.
            # HINT: use torch.distributions.Categorical to define the distribution.
            dist = torch.distributions.Categorical(logits=self.logits_net(obs))
        else:
            # DONE: define the forward pass for a policy with a continuous action space.
            # HINT: use torch.distributions.Normal to define the distribution.
            dist = torch.distributions.Normal(
                loc=self.mean_net(obs),
                scale=torch.exp(self.logstd),
            )

        return dist

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        assert obs.ndim == 2
        assert advantages.ndim == 1
        assert obs.shape[0] == actions.shape[0] == advantages.shape[0]

        obs_t = ptu.from_numpy(obs)
        actions_t = ptu.from_numpy(actions)
        advantages_t = ptu.from_numpy(advantages)

        # DONE: implement the policy gradient actor update.
        # HINT: don't forget to do `self.optimizer.step()`!
        self.optimizer.zero_grad()
        dist = self.forward(obs_t)
        logp = dist.log_prob(actions_t)
        if not self.discrete:
            logp = logp.sum(dim=-1)
        loss = -(logp * advantages_t).mean()
        loss.backward()
        self.optimizer.step()

        return {
            "Actor Loss": ptu.to_numpy(loss),
        }

    def ppo_update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        old_logp: np.ndarray,
        ppo_cliprange: float = 0.2,
    ) -> dict:
        """Implements the policy gradient actor update."""
        assert obs.ndim == 2
        assert advantages.ndim == 1
        assert old_logp.ndim == 1
        assert advantages.shape == old_logp.shape

        obs_t = ptu.from_numpy(obs)
        actions_t = ptu.from_numpy(actions)
        advantages_t = ptu.from_numpy(advantages)
        old_logp_t = ptu.from_numpy(old_logp)

        # DONE: Implement the ppo update.
        # HINT: calculate logp first, and then caculate ratio and clipped loss.
        # HINT: ratio is the exponential of the difference between logp and old_logp.
        # HINT: You can use torch.clamp to clip values.
        self.optimizer.zero_grad()
        dist = self.forward(obs_t)
        logp = dist.log_prob(actions_t)
        if not self.discrete:
            logp = logp.sum(dim=-1)
        ratio = torch.exp(logp - old_logp_t)
        loss = -(
            torch.min(
                ratio * advantages_t,
                torch.clamp(ratio, 1 - ppo_cliprange, 1 + ppo_cliprange) * advantages_t,
            )
        ).mean()
        loss.backward()
        self.optimizer.step()

        return {"PPO Loss": ptu.to_numpy(loss)}
