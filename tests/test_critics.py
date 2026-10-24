import inspect

import numpy as np
import pytest
import torch

from cas4160.networks.critics import ValueCritic
from cas4160.infrastructure import pytorch_util as ptu


@pytest.fixture
def critic(network_params):
    return ValueCritic(
        ob_dim=4,
        n_layers=network_params["n_layers"],
        layer_size=network_params["layer_size"],
        learning_rate=network_params["learning_rate"],
    )


class TestValueCriticInit:
    def test_network_exists(self, critic):
        assert critic.network is not None, "critic.network should not be None"

    def test_optimizer_exists(self, critic):
        assert critic.optimizer is not None, "critic.optimizer should not be None"

    def test_network_output_dim(self, critic):
        """Final linear layer should output scalar (size 1)."""
        linear_layers = [m for m in critic.network if isinstance(m, torch.nn.Linear)]
        actual = linear_layers[-1].out_features
        assert actual == 1, (
            f"expected final Linear layer out_features=1 (scalar value), got {actual}"
        )


class TestValueCriticForward:
    def test_forward_callable(self, critic):
        obs = torch.randn(8, 4)
        result = critic(obs).squeeze()
        # TODO: once forward is implemented, result should be shape (8,)
        assert result.shape == (8,), (
            f"expected forward output shape (8,), got {result.shape}"
        )

    def test_forward_signature(self, critic):
        sig = inspect.signature(critic.forward)
        assert "obs" in sig.parameters, (
            f"'obs' not in forward() params, got: {list(sig.parameters.keys())}"
        )

    def test_forward_output_shape(self, critic):
        """forward() should return shape (batch_size,) — a scalar value per observation."""
        obs = torch.randn(8, 4)
        result = critic(obs).squeeze()
        assert result.shape == (8,), (
            f"expected forward output shape (8,), got {result.shape}"
        )

    def test_forward_single_obs(self, critic):
        """forward() should handle a single observation (batch_size=1)."""
        obs = torch.randn(1, 4)
        result = critic(obs).squeeze(dim=0)
        assert result.shape == (1,), (
            f"expected forward output shape (1,), got {result.shape}"
        )

    def test_forward_is_differentiable(self, critic):
        """forward() output should be differentiable."""
        obs = torch.randn(4, 4)
        result = critic(obs)
        loss = result.sum()
        loss.backward()
        has_grad = any(p.grad is not None for p in critic.parameters())
        assert has_grad, "forward output should be differentiable"


class TestValueCriticUpdate:
    def test_update_signature(self, critic):
        sig = inspect.signature(critic.update)
        params = list(sig.parameters.keys())
        assert "obs" in params, f"'obs' not in update() params, got: {params}"
        assert "q_values" in params, f"'q_values' not in update() params, got: {params}"

    def test_update_returns_baseline_loss(self, critic):
        """update() should return dict with 'Baseline Loss'."""
        obs = np.random.randn(10, 4).astype(np.float32)
        q_values = np.random.randn(10).astype(np.float32)
        result = critic.update(obs, q_values)
        assert isinstance(result, dict), f"expected dict, got {type(result).__name__}"
        assert "Baseline Loss" in result, (
            f"expected 'Baseline Loss' key, got {list(result.keys())}"
        )
        assert np.isfinite(result["Baseline Loss"]), (
            f"Baseline Loss should be finite, got {result['Baseline Loss']}"
        )

    def test_update_changes_params(self, critic):
        """After update, critic parameters should change."""
        obs = np.random.randn(10, 4).astype(np.float32)
        q_values = np.random.randn(10).astype(np.float32)
        params_before = [p.clone() for p in critic.parameters()]
        critic.update(obs, q_values)
        params_after = list(critic.parameters())
        changed = any(
            not torch.equal(before, after)
            for before, after in zip(params_before, params_after)
        )
        assert changed, "update() should modify at least some parameters"

    def test_update_validates_dimensions(self, critic):
        """update() should raise on wrong input dimensions."""
        obs = np.random.randn(10, 4).astype(np.float32)
        q_values = np.random.randn(10, 1).astype(np.float32)  # should be 1D
        with pytest.raises(AssertionError):
            critic.update(obs, q_values)
