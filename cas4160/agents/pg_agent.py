from typing import Optional, Sequence
import numpy as np
import numpy.typing as npt
import torch
from typing import SupportsFloat


from cas4160.networks.policies import MLPPolicyPG
from cas4160.networks.critics import ValueCritic
from cas4160.infrastructure import pytorch_util as ptu
from torch import nn


class PGAgent(nn.Module):
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        gamma: float,
        learning_rate: float,
        use_baseline: bool,
        use_reward_to_go: bool,
        baseline_learning_rate: Optional[float],
        baseline_gradient_steps: Optional[int],
        gae_lambda: Optional[float],
        normalize_advantages: bool,
        use_ppo: bool = False,
        n_ppo_epochs: int = 4,
        n_ppo_minibatches: int = 4,
        ppo_cliprange: float = 0.2,
    ):
        super().__init__()

        # create the actor (policy) network
        self.actor = MLPPolicyPG(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate
        )

        # create the critic (baseline) network, if needed
        if use_baseline:
            assert baseline_learning_rate is not None, (
                "Must specify a baseline learning rate if using a baseline."
            )
            self.critic = ValueCritic(
                ob_dim, n_layers, layer_size, baseline_learning_rate
            )
            self.baseline_gradient_steps = baseline_gradient_steps
        else:
            self.critic = None

        # other agent parameters
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

        self.use_ppo = use_ppo
        self.n_ppo_epochs = n_ppo_epochs if self.use_ppo else None
        self.n_ppo_minibatches = n_ppo_minibatches if use_ppo else None
        self.ppo_cliprange = ppo_cliprange if use_ppo else None

    def update(
        self,
        obs: Sequence[np.ndarray],
        actions: Sequence[np.ndarray],
        rewards: Sequence[np.ndarray],
        terminals: Sequence[np.ndarray],
    ) -> dict:
        """The train step for PG involves updating its actor using the given observations/actions and the calculated
        qvals/advantages that come from the seen rewards.

        Each input is a list of NumPy arrays, where each array corresponds to a single trajectory. The batch size is the
        total number of samples across all trajectories (i.e. the sum of the lengths of all the arrays).
        """

        assert all(ob.ndim == 2 for ob in obs)
        assert all(reward.ndim == 1 for reward in rewards)
        assert all(terminal.ndim == 1 for terminal in terminals)

        # step 1: calculate Q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards)

        # DONE: flatten the lists of arrays into single arrays, so that the rest of the code can be written in a vectorized
        # way. obs, actions, rewards, terminals, and q_values should all be arrays with a leading dimension of `batch_size`
        # beyond this point.
        # HINT: the sum of the lengths of all the arrays is `batch_size`.
        obs_flat: np.ndarray = np.concatenate(obs, axis=0)
        actions_flat: np.ndarray = np.concatenate(actions, axis=0)
        rewards_flat: np.ndarray = np.concatenate(rewards, axis=0)
        terminals_flat: np.ndarray = np.concatenate(terminals, axis=0)
        q_values_flat: np.ndarray = np.concatenate(q_values, axis=0)

        # step 2: calculate advantages from Q values
        assert q_values_flat.ndim == 1
        advantages: np.ndarray = self._estimate_advantage(
            obs_flat, rewards_flat, q_values_flat, terminals_flat
        )

        assert advantages.ndim == 1
        # step 3: use all datapoints (s_t, a_t, adv_t) to update the PG actor/policy
        if not self.use_ppo:
            # DONE: normalize the advantages to have a mean of zero and a standard deviation of one within the batch
            if self.normalize_advantages:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

            # DONE: update the PG actor/policy network once using the advantages
            info: dict = self.actor.update(obs_flat, actions_flat, advantages)

            if self.critic is not None:
                # DONE: update the critic for `baseline_gradient_steps` times
                assert self.baseline_gradient_steps is not None, (
                    "Must specify baseline_gradient_steps if using a baseline."
                )
                for _ in range(self.baseline_gradient_steps):
                    critic_info: dict = self.critic.update(obs_flat, q_values_flat)

                info.update(critic_info)  # pyright: ignore
        else:
            # skip this part until you implement PPO
            # for ppo updates, we need to calculate the log probabilities of old policies
            logp: np.ndarray = self._calculate_log_probs(obs_flat, actions_flat)
            assert logp.ndim == 1

            # gradient steps on minibatches
            n_batch = len(obs_flat)
            inds = np.arange(n_batch)
            for _ in range(self.n_ppo_epochs):  # pyright: ignore
                np.random.shuffle(inds)
                # calculate minibatch size to divide a batch to `n_ppo_minibatches` minibatches
                minibatch_size = (
                    n_batch + (self.n_ppo_minibatches - 1)  # pyright: ignore
                ) // self.n_ppo_minibatches  # pyright: ignore
                for start in range(0, n_batch, minibatch_size):
                    end = start + minibatch_size
                    obs_slice, actions_slice, advantages_slice, logp_slice = (
                        arr[inds[start:end]]
                        for arr in (obs_flat, actions_flat, advantages, logp)
                    )

                    # DONE: normalize `advantages_slice`` to have a mean of zero and a standard deviation of one within the batch
                    if self.normalize_advantages:
                        advantages_slice = (
                            advantages_slice - advantages_slice.mean()
                        ) / (advantages_slice.std() + 1e-8)

                    # DONE: update the PG actor/policy with PPO objective
                    # HINT: call self.actor.ppo_update
                    info: dict = self.actor.ppo_update(
                        obs_slice,
                        actions_slice,
                        advantages_slice,
                        logp_slice,
                        self.ppo_cliprange,  # pyright: ignore
                    )

            assert self.critic is not None, "PPO requires a critic for calculating GAE."
            # TODO: update the critic for `baseline_gradient_steps` times
            for _ in range(self.baseline_gradient_steps):  # pyright: ignore
                critic_info = self.critic.update(obs_flat, q_values_flat)

            info.update(critic_info)  # pyright: ignore
        return info  # pyright: ignore

    def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """Monte Carlo estimation of the Q function."""

        assert all(reward.ndim == 1 for reward in rewards)

        if not self.use_reward_to_go:
            # Case 1: in trajectory-based PG, we ignore the timestep and instead use the discounted return for the entire
            # trajectory at each point.
            # In other words: Q(s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}
            # DONE: use the helper function self._discounted_return to calculate the Q-values

            q_values = [self._discounted_return(reward) for reward in rewards]
        else:
            # Case 2: in reward-to-go PG, we only use the rewards after timestep t to estimate the Q-value for (s_t, a_t).
            # In other words: Q(s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
            # DONE: use the helper function self._discounted_reward_to_go to calculate the Q-values
            q_values = [self._discounted_reward_to_go(reward) for reward in rewards]

        return q_values

    def _estimate_advantage(
        self,
        obs: np.ndarray,
        rewards: np.ndarray,
        q_values: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        """Computes advantages by (possibly) subtracting a value baseline from the estimated Q-values."""

        assert obs.ndim == 2

        if self.critic is None:
            # DONE: if no baseline, then what are the advantages?
            advantages = q_values
        else:
            # DONE: run the critic and use it as a baseline
            values = self.critic(ptu.from_numpy(obs)).detach().cpu().numpy().squeeze()

            assert values.ndim == 1, "Critic output should have shape (batch_size,)"
            assert values.shape == q_values.shape, (
                "Values and Q-values should have the same shape"
            )

            if self.gae_lambda is None:
                # DONE: if using a baseline, but not GAE, what are the advantages?
                advantages = q_values - values
            else:
                # DONE: implement GAE
                batch_size = obs.shape[0]

                # HINT: append a dummy T+1 value for simpler recursive calculation
                values = np.append(values, [0])
                advantages = np.zeros(batch_size + 1)

                # HINT: calculating `deltas` as in the GAE formula first would be useful.
                # HINT2: handle edge cases by using `terminals`. You can multiply (1 - terminals) to the value of the next state
                # to handle this.
                deltas = (
                    rewards + self.gamma * values[1:] * (1 - terminals) - values[:-1]
                )

                for i in reversed(range(batch_size)):
                    # DONE: recursively compute advantage estimates starting from timestep T.
                    # HINT: use terminals to handle edge cases. terminals[i] is 1 if there isn't a next state in its
                    # trajectory, and 0 otherwise.
                    advantages[i] = deltas[
                        i
                    ] + self.gamma * self.gae_lambda * advantages[i + 1] * (
                        1 - terminals[i]
                    )

                # remove dummy advantage
                advantages = advantages[:-1]

        return advantages

    def _discounted_return(
        self, rewards: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns
        a list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}

        Note that all entries of the output list should be the exact same because each sum is from 0 to T (and doesn't
        involve t)!

        Example:
        ```python
        # assume gamma = 0.99
        rewards = np.array([1., 2., 3.])
        total_discounted_return = agent._discounted_return(rewards)
        print(total_discounted_return)
        ```

        Output:
        ```
        np.array([5.9203, 5.9203, 5.9203])
        ```
        """
        assert rewards.ndim == 1
        # DONE: calculate discounted return using the above formula
        ret: np.ndarray = (self.gamma ** np.arange(len(rewards)) @ rewards).repeat(
            len(rewards)
        )

        assert rewards.shape == ret.shape
        return ret

    def _discounted_reward_to_go(
        self, rewards: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns a list where the entry
        in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}.

        Example:
        ```python
        # assume gamma = 0.99
        rewards = np.array([1., 2., 3.])
        total_discounted_return = agent._discounted_reward_to_go(rewards)
        print(total_discounted_return)
        ```

        Output:
        ```
        np.array([5.9203, 4.97, 3.])
        ```
        """
        assert rewards.ndim == 1
        # DONE: calculate discounted reward to go using the above formula
        discount_factors = self.gamma ** np.arange(len(rewards))
        weighted_rewards = discount_factors * rewards
        ret = (weighted_rewards[::-1].cumsum())[::-1] / discount_factors

        assert rewards.shape == ret.shape
        return ret

    def _calculate_log_probs(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        """
        Helper function to calculate the log probabilities of the actions taken by the actor.
        """

        assert obs.ndim == 2
        # TODO: calculate the log probabilities
        # HINT: self.actor outputs a distribution object, which has a method log_prob that takes in the actions
        dist = self.actor(ptu.from_numpy(obs))
        if self.actor.discrete:
            logp = dist.log_prob(ptu.from_numpy(actions)).detach().cpu().numpy()
        else:
            logp = (
                dist.log_prob(ptu.from_numpy(actions))
                .detach()
                .cpu()
                .numpy()
                .sum(axis=-1)
            )

        assert logp.ndim == 1 and logp.shape[0] == obs.shape[0]
        return logp
