from unittest.mock import patch

import numpy as np
import pytest
import torch

from cas4160.agents.pg_agent import PGAgent
from cas4160.infrastructure import pytorch_util as ptu


@pytest.fixture
def agent_no_baseline():
    return PGAgent(
        ob_dim=4,
        ac_dim=2,
        discrete=True,
        n_layers=2,
        layer_size=64,
        gamma=0.99,
        learning_rate=5e-3,
        use_baseline=False,
        use_reward_to_go=False,
        baseline_learning_rate=None,
        baseline_gradient_steps=None,
        gae_lambda=None,
        normalize_advantages=False,
    )


@pytest.fixture
def agent_with_baseline():
    return PGAgent(
        ob_dim=4,
        ac_dim=2,
        discrete=True,
        n_layers=2,
        layer_size=64,
        gamma=0.99,
        learning_rate=5e-3,
        use_baseline=True,
        use_reward_to_go=True,
        baseline_learning_rate=5e-3,
        baseline_gradient_steps=5,
        gae_lambda=None,
        normalize_advantages=True,
    )


@pytest.fixture
def agent_gae():
    return PGAgent(
        ob_dim=4,
        ac_dim=2,
        discrete=True,
        n_layers=2,
        layer_size=64,
        gamma=0.99,
        learning_rate=5e-3,
        use_baseline=True,
        use_reward_to_go=True,
        baseline_learning_rate=5e-3,
        baseline_gradient_steps=5,
        gae_lambda=0.95,
        normalize_advantages=True,
    )


@pytest.fixture
def agent_ppo():
    return PGAgent(
        ob_dim=4,
        ac_dim=2,
        discrete=True,
        n_layers=2,
        layer_size=64,
        gamma=0.99,
        learning_rate=5e-3,
        use_baseline=True,
        use_reward_to_go=True,
        baseline_learning_rate=5e-3,
        baseline_gradient_steps=5,
        gae_lambda=0.95,
        normalize_advantages=True,
        use_ppo=True,
        n_ppo_epochs=4,
        n_ppo_minibatches=4,
        ppo_cliprange=0.2,
    )


class TestPGAgentInit:
    def test_actor_created(self, agent_no_baseline):
        assert agent_no_baseline.actor is not None, "agent.actor should not be None"

    def test_no_critic_without_baseline(self, agent_no_baseline):
        assert agent_no_baseline.critic is None, (
            f"expected critic=None without baseline, got {type(agent_no_baseline.critic).__name__}"
        )

    def test_critic_with_baseline(self, agent_with_baseline):
        assert agent_with_baseline.critic is not None, (
            "expected critic to be created when use_baseline=True"
        )

    def test_ppo_params_set(self, agent_ppo):
        assert agent_ppo.use_ppo is True, (
            f"expected use_ppo=True, got {agent_ppo.use_ppo}"
        )
        assert agent_ppo.n_ppo_epochs == 4, (
            f"expected n_ppo_epochs=4, got {agent_ppo.n_ppo_epochs}"
        )
        assert agent_ppo.n_ppo_minibatches == 4, (
            f"expected n_ppo_minibatches=4, got {agent_ppo.n_ppo_minibatches}"
        )
        assert agent_ppo.ppo_cliprange == 0.2, (
            f"expected ppo_cliprange=0.2, got {agent_ppo.ppo_cliprange}"
        )

    def test_ppo_params_none_when_off(self, agent_no_baseline):
        assert agent_no_baseline.n_ppo_epochs is None, (
            f"expected n_ppo_epochs=None when PPO off, got {agent_no_baseline.n_ppo_epochs}"
        )
        assert agent_no_baseline.n_ppo_minibatches is None, (
            f"expected n_ppo_minibatches=None when PPO off, got {agent_no_baseline.n_ppo_minibatches}"
        )
        assert agent_no_baseline.ppo_cliprange is None, (
            f"expected ppo_cliprange=None when PPO off, got {agent_no_baseline.ppo_cliprange}"
        )

    def test_gamma_stored(self, agent_no_baseline):
        assert agent_no_baseline.gamma == 0.99, (
            f"expected gamma=0.99, got {agent_no_baseline.gamma}"
        )

    def test_baseline_requires_lr(self):
        with pytest.raises(AssertionError, match="baseline learning rate"):
            PGAgent(
                ob_dim=4,
                ac_dim=2,
                discrete=True,
                n_layers=2,
                layer_size=64,
                gamma=0.99,
                learning_rate=5e-3,
                use_baseline=True,
                use_reward_to_go=False,
                baseline_learning_rate=None,
                baseline_gradient_steps=5,
                gae_lambda=None,
                normalize_advantages=False,
            )


class TestDiscountedReturn:
    def test_constant_rewards_gamma1(self, agent_no_baseline):
        agent_no_baseline.gamma = 1.0
        rewards = np.array([1.0, 1.0, 1.0])
        result = agent_no_baseline._discounted_return(rewards)
        expected = [3.0, 3.0, 3.0]
        np.testing.assert_allclose(
            result,
            expected,
            err_msg=f"gamma=1.0, rewards={rewards.tolist()}: expected {expected}, got {result.tolist()}",
        )

    def test_discounted_return_example(self, agent_no_baseline):
        """Match docstring example: gamma=0.99, rewards=[1,2,3] -> 5.9203 for all."""
        agent_no_baseline.gamma = 0.99
        rewards = np.array([1.0, 2.0, 3.0])
        result = agent_no_baseline._discounted_return(rewards)
        expected_val = 1.0 + 0.99 * 2.0 + 0.99**2 * 3.0
        expected = [expected_val, expected_val, expected_val]
        np.testing.assert_allclose(
            result,
            expected,
            atol=1e-4,
            err_msg=f"gamma=0.99, rewards=[1,2,3]: expected all {expected_val:.4f}, got {result.tolist()}",
        )

    def test_single_reward(self, agent_no_baseline):
        rewards = np.array([5.0])
        result = agent_no_baseline._discounted_return(rewards)
        np.testing.assert_allclose(
            result,
            [5.0],
            err_msg=f"single reward=5.0: expected [5.0], got {result.tolist()}",
        )

    def test_output_shape_matches_input(self, agent_no_baseline):
        rewards = np.random.randn(10)
        result = agent_no_baseline._discounted_return(rewards)
        assert result.shape == rewards.shape, (
            f"expected output shape {rewards.shape}, got {result.shape}"
        )

    def test_all_entries_equal(self, agent_no_baseline):
        """All entries should be the same (discounted sum from 0 to T)."""
        rewards = np.array([1.0, 2.0, 3.0, 4.0])
        result = agent_no_baseline._discounted_return(rewards)
        assert np.all(result == result[0]), (
            f"all entries should equal {result[0]}, got {result.tolist()}"
        )


class TestDiscountedRewardToGo:
    def test_constant_rewards_gamma1(self, agent_no_baseline):
        agent_no_baseline.gamma = 1.0
        rewards = np.array([1.0, 1.0, 1.0])
        result = agent_no_baseline._discounted_reward_to_go(rewards)
        expected = [3.0, 2.0, 1.0]
        np.testing.assert_allclose(
            result,
            expected,
            err_msg=f"gamma=1.0, rewards={rewards.tolist()}: expected {expected}, got {result.tolist()}",
        )

    def test_reward_to_go_example(self, agent_no_baseline):
        """Match docstring example: gamma=0.99, rewards=[1,2,3].
        Q(s_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}, note the (t'-t) exponent."""
        agent_no_baseline.gamma = 0.99
        rewards = np.array([1.0, 2.0, 3.0])
        result = agent_no_baseline._discounted_reward_to_go(rewards)
        expected = [
            1.0 + 0.99 * 2.0 + 0.99**2 * 3.0,
            2.0 + 0.99 * 3.0,
            3.0,
        ]
        np.testing.assert_allclose(
            result,
            expected,
            atol=1e-4,
            err_msg=(
                f"gamma=0.99, rewards=[1,2,3]: expected {expected}, got {result.tolist()}. "
                f"The exponent should be gamma^(t'-t), not gamma^(t')."
            ),
        )

    def test_last_entry_equals_last_reward(self, agent_no_baseline):
        rewards = np.array([1.0, 2.0, 3.0])
        result = agent_no_baseline._discounted_reward_to_go(rewards)
        np.testing.assert_allclose(
            result[-1],
            rewards[-1],
            err_msg=(
                f"last entry should equal last reward {rewards[-1]}, "
                f"got {result[-1]} (gamma^T * r_T instead of r_T)"
            ),
        )

    def test_monotonically_related(self, agent_no_baseline):
        """With positive rewards and gamma<1, reward-to-go should be non-increasing
        when all rewards are equal."""
        agent_no_baseline.gamma = 0.99
        rewards = np.ones(5)
        result = agent_no_baseline._discounted_reward_to_go(rewards)
        for i in range(len(result) - 1):
            assert result[i] >= result[i + 1], (
                f"result[{i}]={result[i]} < result[{i + 1}]={result[i + 1]}, "
                f"but reward-to-go should be non-increasing for equal positive rewards. "
                f"Full result: {result.tolist()}"
            )

    def test_output_shape(self, agent_no_baseline):
        rewards = np.random.randn(7)
        result = agent_no_baseline._discounted_reward_to_go(rewards)
        assert result.shape == rewards.shape, (
            f"expected output shape {rewards.shape}, got {result.shape}"
        )


class TestCalculateQVals:
    def test_no_reward_to_go(self, agent_no_baseline):
        """Without reward-to-go, each trajectory gets same Q for all timesteps."""
        agent_no_baseline.use_reward_to_go = False
        rewards = [np.array([1.0, 2.0, 3.0])]
        result = agent_no_baseline._calculate_q_vals(rewards)
        assert len(result) == 1, f"expected 1 trajectory in result, got {len(result)}"
        assert np.all(result[0] == result[0][0]), (
            f"without reward-to-go, all Q-values should be equal, got {result[0].tolist()}"
        )

    def test_with_reward_to_go(self, agent_no_baseline):
        agent_no_baseline.use_reward_to_go = True
        rewards = [np.array([1.0, 2.0, 3.0])]
        result = agent_no_baseline._calculate_q_vals(rewards)
        assert len(result) == 1, f"expected 1 trajectory in result, got {len(result)}"
        assert result[0].shape == rewards[0].shape, (
            f"expected Q-values shape {rewards[0].shape}, got {result[0].shape}"
        )

    def test_multiple_trajectories(self, agent_no_baseline):
        agent_no_baseline.use_reward_to_go = False
        rewards = [np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])]
        result = agent_no_baseline._calculate_q_vals(rewards)
        assert len(result) == 2, f"expected 2 trajectories in result, got {len(result)}"
        assert result[0].shape == (2,), (
            f"expected first trajectory Q-values shape (2,), got {result[0].shape}"
        )
        assert result[1].shape == (3,), (
            f"expected second trajectory Q-values shape (3,), got {result[1].shape}"
        )


class TestEstimateAdvantage:
    def test_no_baseline_returns_qvals(self, agent_no_baseline):
        obs = np.random.randn(5, 4).astype(np.float32)
        rewards = np.ones(5, dtype=np.float32)
        q_values = np.array([3.0, 2.0, 1.0, 0.5, 0.1], dtype=np.float32)
        terminals = np.zeros(5, dtype=np.float32)
        result = agent_no_baseline._estimate_advantage(
            obs, rewards, q_values, terminals
        )
        np.testing.assert_allclose(
            result,
            q_values,
            err_msg=(
                f"without baseline, advantages should equal Q-values. "
                f"expected {q_values.tolist()}, got {result.tolist()}"
            ),
        )

    @pytest.mark.xfail(reason="TODO: critic forward not yet implemented")
    def test_baseline_subtracts_values(self, agent_with_baseline):
        """With baseline (no GAE), advantages = Q - V(s)."""
        obs = np.random.randn(5, 4).astype(np.float32)
        rewards = np.ones(5, dtype=np.float32)
        q_values = np.array([3.0, 2.0, 1.0, 0.5, 0.1], dtype=np.float32)
        terminals = np.zeros(5, dtype=np.float32)

        # Get critic values for comparison
        with torch.no_grad():
            values = agent_with_baseline.critic(
                ptu.from_numpy(obs)
            ).squeeze().cpu().numpy()

        result = agent_with_baseline._estimate_advantage(
            obs, rewards, q_values, terminals
        )
        expected = q_values - values
        np.testing.assert_allclose(
            result,
            expected,
            atol=1e-5,
            err_msg="with baseline (no GAE), advantages should be Q - V(s)",
        )

    @pytest.mark.xfail(reason="TODO: critic forward not yet implemented")
    def test_baseline_advantage_shape(self, agent_with_baseline):
        obs = np.random.randn(8, 4).astype(np.float32)
        rewards = np.ones(8, dtype=np.float32)
        q_values = np.random.randn(8).astype(np.float32)
        terminals = np.zeros(8, dtype=np.float32)
        result = agent_with_baseline._estimate_advantage(
            obs, rewards, q_values, terminals
        )
        assert result.shape == (8,), (
            f"expected advantage shape (8,), got {result.shape}"
        )

    @pytest.mark.xfail(reason="TODO: critic forward not yet implemented")
    def test_gae_output_shape(self, agent_gae):
        """GAE should return advantages with the correct shape."""
        obs = np.random.randn(10, 4).astype(np.float32)
        rewards = np.ones(10, dtype=np.float32)
        q_values = np.random.randn(10).astype(np.float32)
        terminals = np.zeros(10, dtype=np.float32)
        result = agent_gae._estimate_advantage(obs, rewards, q_values, terminals)
        assert result.shape == (10,), (
            f"expected GAE advantage shape (10,), got {result.shape}"
        )

    @pytest.mark.xfail(reason="TODO: critic forward not yet implemented")
    def test_gae_terminal_resets(self, agent_gae):
        """GAE advantages should not propagate across episode boundaries."""
        obs = np.random.randn(6, 4).astype(np.float32)
        rewards = np.ones(6, dtype=np.float32)
        q_values = np.random.randn(6).astype(np.float32)
        # Terminal at step 2 means episode ends there
        terminals = np.array([0, 0, 1, 0, 0, 1], dtype=np.float32)
        result = agent_gae._estimate_advantage(obs, rewards, q_values, terminals)
        assert result.shape == (6,), (
            f"expected advantage shape (6,), got {result.shape}"
        )
        # Result should be finite (no NaN/Inf from broken terminal handling)
        assert np.all(np.isfinite(result)), (
            f"GAE advantages should be finite, got {result.tolist()}"
        )

    @pytest.mark.xfail(reason="TODO: critic forward not yet implemented")
    def test_gae_lambda_zero_equals_td(self, agent_gae):
        """When gae_lambda=0, GAE should reduce to TD residuals: delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)."""
        agent_gae.gae_lambda = 0.0
        obs = np.random.randn(5, 4).astype(np.float32)
        rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        q_values = np.random.randn(5).astype(np.float32)
        terminals = np.zeros(5, dtype=np.float32)

        with torch.no_grad():
            values = agent_gae.critic(
                ptu.from_numpy(obs)
            ).squeeze().cpu().numpy()

        result = agent_gae._estimate_advantage(obs, rewards, q_values, terminals)
        # With lambda=0, advantage[t] = delta[t] = r_t + gamma * V(s_{t+1}) - V(s_t)
        values_ext = np.append(values, [0])
        expected_deltas = rewards + agent_gae.gamma * values_ext[1:] * (1 - terminals) - values_ext[:-1]
        np.testing.assert_allclose(
            result,
            expected_deltas,
            atol=1e-5,
            err_msg="with gae_lambda=0, advantages should equal TD residuals",
        )


class TestCalculateLogProbs:
    @pytest.mark.xfail(reason="TODO: _calculate_log_probs not yet implemented")
    def test_output_shape(self, agent_no_baseline):
        obs = np.random.randn(10, 4).astype(np.float32)
        actions = np.random.randint(0, 2, size=(10,)).astype(np.float32)
        result = agent_no_baseline._calculate_log_probs(obs, actions)
        assert result.shape == (10,), (
            f"expected log_probs shape (10,), got {result.shape}"
        )

    @pytest.mark.xfail(reason="TODO: _calculate_log_probs not yet implemented")
    def test_log_probs_are_negative(self, agent_no_baseline):
        """Log probabilities should be <= 0."""
        obs = np.random.randn(10, 4).astype(np.float32)
        actions = np.random.randint(0, 2, size=(10,)).astype(np.float32)
        result = agent_no_baseline._calculate_log_probs(obs, actions)
        assert np.all(result <= 0.0 + 1e-6), (
            f"log probabilities should be <= 0, got max={result.max()}"
        )

    @pytest.mark.xfail(reason="TODO: _calculate_log_probs not yet implemented")
    def test_returns_numpy(self, agent_no_baseline):
        obs = np.random.randn(5, 4).astype(np.float32)
        actions = np.random.randint(0, 2, size=(5,)).astype(np.float32)
        result = agent_no_baseline._calculate_log_probs(obs, actions)
        assert isinstance(result, np.ndarray), (
            f"expected np.ndarray, got {type(result).__name__}"
        )


class TestPGAgentUpdate:
    @pytest.mark.xfail(reason="TODO: actor update not yet implemented")
    def test_update_returns_dict(self, agent_no_baseline):
        """update() should return a dict with loss info."""
        obs = [np.random.randn(10, 4).astype(np.float32)]
        actions = [np.random.randint(0, 2, size=(10,)).astype(np.float32)]
        rewards = [np.ones(10, dtype=np.float32)]
        terminals = [np.zeros(10, dtype=np.float32)]
        result = agent_no_baseline.update(obs, actions, rewards, terminals)
        assert isinstance(result, dict), (
            f"expected dict, got {type(result).__name__}"
        )
        assert "Actor Loss" in result, (
            f"expected 'Actor Loss' key in result, got keys: {list(result.keys())}"
        )

    @pytest.mark.xfail(reason="TODO: actor update not yet implemented")
    def test_update_with_baseline_returns_critic_loss(self, agent_with_baseline):
        """update() with baseline should also return Baseline Loss."""
        obs = [np.random.randn(10, 4).astype(np.float32)]
        actions = [np.random.randint(0, 2, size=(10,)).astype(np.float32)]
        rewards = [np.ones(10, dtype=np.float32)]
        terminals = [np.zeros(10, dtype=np.float32)]
        result = agent_with_baseline.update(obs, actions, rewards, terminals)
        assert "Baseline Loss" in result, (
            f"expected 'Baseline Loss' key in result, got keys: {list(result.keys())}"
        )

    @pytest.mark.xfail(reason="TODO: actor update not yet implemented")
    def test_update_multiple_trajectories(self, agent_no_baseline):
        """update() should handle multiple trajectories."""
        obs = [
            np.random.randn(5, 4).astype(np.float32),
            np.random.randn(8, 4).astype(np.float32),
        ]
        actions = [
            np.random.randint(0, 2, size=(5,)).astype(np.float32),
            np.random.randint(0, 2, size=(8,)).astype(np.float32),
        ]
        rewards = [np.ones(5, dtype=np.float32), np.ones(8, dtype=np.float32)]
        terminals = [np.zeros(5, dtype=np.float32), np.zeros(8, dtype=np.float32)]
        result = agent_no_baseline.update(obs, actions, rewards, terminals)
        assert isinstance(result, dict)


class TestAdvantageNormalization:
    def test_normalize_zero_mean_unit_std(self):
        """After normalization, advantages should have ~zero mean and ~unit std."""
        # This tests the normalization logic conceptually
        advantages = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        np.testing.assert_allclose(
            normalized.mean(), 0.0, atol=1e-6,
            err_msg="normalized advantages should have zero mean",
        )
        np.testing.assert_allclose(
            normalized.std(), 1.0, atol=1e-6,
            err_msg="normalized advantages should have unit std",
        )
