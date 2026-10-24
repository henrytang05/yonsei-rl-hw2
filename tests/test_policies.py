import inspect

import numpy as np
import pytest
import torch
from torch import distributions

from cas4160.networks.policies import MLPPolicy, MLPPolicyPG
from cas4160.infrastructure import pytorch_util as ptu


@pytest.fixture
def discrete_policy(discrete_env_dims, network_params):
    return MLPPolicyPG(
        ac_dim=discrete_env_dims["ac_dim"],
        ob_dim=discrete_env_dims["ob_dim"],
        discrete=True,
        n_layers=network_params["n_layers"],
        layer_size=network_params["layer_size"],
        learning_rate=network_params["learning_rate"],
    )


@pytest.fixture
def continuous_policy(continuous_env_dims, network_params):
    return MLPPolicyPG(
        ac_dim=continuous_env_dims["ac_dim"],
        ob_dim=continuous_env_dims["ob_dim"],
        discrete=False,
        n_layers=network_params["n_layers"],
        layer_size=network_params["layer_size"],
        learning_rate=network_params["learning_rate"],
    )


class TestMLPPolicyInit:
    def test_discrete_has_logits_net(self, discrete_policy):
        assert hasattr(discrete_policy, "logits_net"), (
            "discrete policy should have 'logits_net' attribute"
        )
        assert not hasattr(discrete_policy, "mean_net"), (
            "discrete policy should NOT have 'mean_net' attribute"
        )

    def test_continuous_has_mean_net_and_logstd(self, continuous_policy):
        assert hasattr(continuous_policy, "mean_net"), (
            "continuous policy should have 'mean_net' attribute"
        )
        assert hasattr(continuous_policy, "logstd"), (
            "continuous policy should have 'logstd' attribute"
        )
        assert continuous_policy.logstd.shape == (3,), (
            f"expected logstd shape (3,), got {continuous_policy.logstd.shape}"
        )

    def test_continuous_logstd_init_zeros(self, continuous_policy):
        actual = ptu.to_numpy(continuous_policy.logstd)
        np.testing.assert_allclose(
            actual, np.zeros(3), atol=1e-7,
            err_msg=f"logstd should be initialized to zeros, got {actual}",
        )

    def test_optimizer_exists(self, discrete_policy, continuous_policy):
        assert discrete_policy.optimizer is not None, (
            "discrete policy optimizer is None"
        )
        assert continuous_policy.optimizer is not None, (
            "continuous policy optimizer is None"
        )


class TestGetAction:
    @pytest.mark.xfail(reason="TODO: get_action not yet implemented for discrete")
    def test_discrete_action_shape(self, discrete_policy):
        obs = np.random.randn(4).astype(np.float32)
        action = discrete_policy.get_action(obs)
        assert isinstance(action, np.ndarray), (
            f"expected np.ndarray, got {type(action).__name__}"
        )

    def test_continuous_action_shape(self, continuous_policy):
        obs = np.random.randn(8).astype(np.float32)
        action = continuous_policy.get_action(obs)
        assert isinstance(action, np.ndarray), (
            f"expected np.ndarray, got {type(action).__name__}"
        )
        assert action.shape == (3,), (
            f"expected action shape (3,), got {action.shape}"
        )

    @pytest.mark.xfail(reason="TODO: get_action not yet implemented for discrete")
    def test_no_grad_context(self, discrete_policy):
        """get_action should not track gradients."""
        obs = np.random.randn(4).astype(np.float32)
        discrete_policy.get_action(obs)
        for name, param in discrete_policy.named_parameters():
            assert param.grad is None, (
                f"param '{name}' has grad={param.grad}, expected None (no gradient tracking)"
            )


class TestForward:
    """Tests for forward() — these verify structure.
    Once the TODOs are filled in, forward should return a Distribution."""

    def test_discrete_forward_callable(self, discrete_policy):
        obs = torch.randn(8, 4)
        result = discrete_policy(obs)
        # TODO is not implemented yet — result may be None
        # Once implemented, uncomment:
        # assert isinstance(result, distributions.Categorical), (
        #     f"expected Categorical distribution, got {type(result).__name__}"
        # )
        # assert result.batch_shape == (8,), (
        #     f"expected batch_shape (8,), got {result.batch_shape}"
        # )

    def test_continuous_forward_callable(self, continuous_policy):
        obs = torch.randn(8, 8)
        result = continuous_policy(obs)
        # Once implemented, uncomment:
        # assert isinstance(result, distributions.Normal), (
        #     f"expected Normal distribution, got {type(result).__name__}"
        # )
        # assert result.batch_shape == (8, 3), (
        #     f"expected batch_shape (8, 3), got {result.batch_shape}"
        # )

    @pytest.mark.xfail(reason="TODO: forward not yet implemented for discrete")
    def test_discrete_forward_returns_categorical(self, discrete_policy):
        obs = torch.randn(8, 4)
        result = discrete_policy(obs)
        assert isinstance(result, distributions.Categorical), (
            f"expected Categorical distribution, got {type(result).__name__}"
        )
        assert result.batch_shape == (8,), (
            f"expected batch_shape (8,), got {result.batch_shape}"
        )

    @pytest.mark.xfail(reason="TODO: forward not yet implemented for continuous")
    def test_continuous_forward_returns_normal(self, continuous_policy):
        obs = torch.randn(8, 8)
        result = continuous_policy(obs)
        assert isinstance(result, distributions.Normal), (
            f"expected Normal distribution, got {type(result).__name__}"
        )
        assert result.batch_shape == (8, 3), (
            f"expected batch_shape (8, 3), got {result.batch_shape}"
        )

    @pytest.mark.xfail(reason="TODO: forward not yet implemented for discrete")
    def test_discrete_forward_is_differentiable(self, discrete_policy):
        obs = torch.randn(8, 4, requires_grad=False)
        dist = discrete_policy(obs)
        sample = dist.sample()
        log_prob = dist.log_prob(sample)
        assert log_prob.shape == (8,), (
            f"expected log_prob shape (8,), got {log_prob.shape}"
        )

    @pytest.mark.xfail(reason="TODO: forward not yet implemented for continuous")
    def test_continuous_forward_is_differentiable(self, continuous_policy):
        obs = torch.randn(8, 8, requires_grad=False)
        dist = continuous_policy(obs)
        sample = dist.sample()
        log_prob = dist.log_prob(sample)
        assert log_prob.shape == (8, 3), (
            f"expected log_prob shape (8, 3), got {log_prob.shape}"
        )


class TestMLPPolicyPGUpdate:
    """Tests for the PG update method. Skeleton tests until TODOs are filled."""

    def test_update_signature(self, discrete_policy):
        """update() should accept obs, actions, advantages."""
        sig = inspect.signature(discrete_policy.update)
        params = list(sig.parameters.keys())
        assert "obs" in params, (
            f"'obs' not in update() params, got: {params}"
        )
        assert "actions" in params, (
            f"'actions' not in update() params, got: {params}"
        )
        assert "advantages" in params, (
            f"'advantages' not in update() params, got: {params}"
        )

    def test_ppo_update_signature(self, discrete_policy):
        sig = inspect.signature(discrete_policy.ppo_update)
        params = list(sig.parameters.keys())
        assert "old_logp" in params, (
            f"'old_logp' not in ppo_update() params, got: {params}"
        )
        assert "ppo_cliprange" in params, (
            f"'ppo_cliprange' not in ppo_update() params, got: {params}"
        )

    @pytest.mark.xfail(reason="TODO: update not yet implemented")
    def test_update_returns_actor_loss(self, discrete_policy):
        """update() should return dict with 'Actor Loss'."""
        obs = np.random.randn(10, 4).astype(np.float32)
        actions = np.random.randint(0, 2, size=(10,)).astype(np.float32)
        advantages = np.random.randn(10).astype(np.float32)
        result = discrete_policy.update(obs, actions, advantages)
        assert isinstance(result, dict), f"expected dict, got {type(result).__name__}"
        assert "Actor Loss" in result, f"expected 'Actor Loss' key, got {list(result.keys())}"
        assert np.isfinite(result["Actor Loss"]), f"Actor Loss should be finite, got {result['Actor Loss']}"

    @pytest.mark.xfail(reason="TODO: update not yet implemented")
    def test_update_changes_params(self, discrete_policy):
        """After update, at least some parameters should have changed."""
        obs = np.random.randn(10, 4).astype(np.float32)
        actions = np.random.randint(0, 2, size=(10,)).astype(np.float32)
        advantages = np.random.randn(10).astype(np.float32)
        params_before = [p.clone() for p in discrete_policy.parameters()]
        discrete_policy.update(obs, actions, advantages)
        params_after = list(discrete_policy.parameters())
        changed = any(
            not torch.equal(before, after)
            for before, after in zip(params_before, params_after)
        )
        assert changed, "update() should modify at least some parameters"

    @pytest.mark.xfail(reason="TODO: update not yet implemented")
    def test_update_validates_shapes(self, discrete_policy):
        """update() should raise on mismatched shapes."""
        obs = np.random.randn(10, 4).astype(np.float32)
        actions = np.random.randint(0, 2, size=(5,)).astype(np.float32)  # wrong size
        advantages = np.random.randn(10).astype(np.float32)
        with pytest.raises(AssertionError):
            discrete_policy.update(obs, actions, advantages)

    @pytest.mark.xfail(reason="TODO: update not yet implemented for continuous")
    def test_continuous_update(self, continuous_policy):
        """update() should work for continuous policies."""
        obs = np.random.randn(10, 8).astype(np.float32)
        actions = np.random.randn(10, 3).astype(np.float32)
        advantages = np.random.randn(10).astype(np.float32)
        result = continuous_policy.update(obs, actions, advantages)
        assert "Actor Loss" in result

    @pytest.mark.xfail(reason="TODO: ppo_update not yet implemented")
    def test_ppo_update_returns_loss(self, discrete_policy):
        """ppo_update() should return dict with 'PPO Loss'."""
        obs = np.random.randn(10, 4).astype(np.float32)
        actions = np.random.randint(0, 2, size=(10,)).astype(np.float32)
        advantages = np.random.randn(10).astype(np.float32)
        old_logp = np.random.randn(10).astype(np.float32)
        result = discrete_policy.ppo_update(obs, actions, advantages, old_logp)
        assert isinstance(result, dict), f"expected dict, got {type(result).__name__}"
        assert "PPO Loss" in result, f"expected 'PPO Loss' key, got {list(result.keys())}"
        assert np.isfinite(result["PPO Loss"]), f"PPO Loss should be finite"

    @pytest.mark.xfail(reason="TODO: ppo_update not yet implemented")
    def test_ppo_update_validates_shapes(self, discrete_policy):
        """ppo_update() should raise on mismatched old_logp/advantages shapes."""
        obs = np.random.randn(10, 4).astype(np.float32)
        actions = np.random.randint(0, 2, size=(10,)).astype(np.float32)
        advantages = np.random.randn(10).astype(np.float32)
        old_logp = np.random.randn(5).astype(np.float32)  # wrong size
        with pytest.raises(AssertionError):
            discrete_policy.ppo_update(obs, actions, advantages, old_logp)
