import numpy as np
import pytest
import gymnasium as gym

from cas4160.infrastructure.utils import (
    rollout_trajectory,
    rollout_trajectories,
    rollout_n_trajectories,
    compute_metrics,
    convert_listofrollouts,
    get_traj_length,
)
from cas4160.networks.policies import MLPPolicyPG
from cas4160.infrastructure import pytorch_util as ptu


@pytest.fixture
def cartpole_env():
    env = gym.make("CartPole-v1")
    yield env
    env.close()


@pytest.fixture
def cartpole_policy(network_params):
    return MLPPolicyPG(
        ac_dim=2,
        ob_dim=4,
        discrete=True,
        n_layers=network_params["n_layers"],
        layer_size=network_params["layer_size"],
        learning_rate=network_params["learning_rate"],
    )


@pytest.mark.xfail(reason="TODO: get_action not yet implemented for discrete")
class TestRolloutTrajectory:
    def test_returns_dict(self, cartpole_env, cartpole_policy):
        traj = rollout_trajectory(cartpole_env, cartpole_policy, max_length=10)
        assert isinstance(traj, dict), (
            f"expected dict, got {type(traj).__name__}"
        )

    def test_required_keys(self, cartpole_env, cartpole_policy):
        traj = rollout_trajectory(cartpole_env, cartpole_policy, max_length=10)
        expected_keys = {"observation", "image_obs", "reward", "action", "next_observation", "terminal"}
        actual_keys = set(traj.keys())
        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys
        assert expected_keys == actual_keys, (
            f"key mismatch — missing: {missing or 'none'}, extra: {extra or 'none'}"
        )

    def test_max_length_respected(self, cartpole_env, cartpole_policy):
        traj = rollout_trajectory(cartpole_env, cartpole_policy, max_length=5)
        length = len(traj["reward"])
        assert length <= 5, (
            f"expected trajectory length <= 5, got {length}"
        )

    def test_obs_shape(self, cartpole_env, cartpole_policy):
        traj = rollout_trajectory(cartpole_env, cartpole_policy, max_length=10)
        assert traj["observation"].ndim == 2, (
            f"expected obs ndim=2, got {traj['observation'].ndim} (shape={traj['observation'].shape})"
        )
        assert traj["observation"].shape[1] == 4, (
            f"expected obs dim=4 (CartPole), got {traj['observation'].shape[1]}"
        )

    def test_terminal_last_true(self, cartpole_env, cartpole_policy):
        traj = rollout_trajectory(cartpole_env, cartpole_policy, max_length=5)
        assert traj["terminal"][-1] == 1.0, (
            f"expected last terminal=1.0, got {traj['terminal'][-1]}. "
            f"Full terminals: {traj['terminal'].tolist()}"
        )

    def test_dtypes(self, cartpole_env, cartpole_policy):
        traj = rollout_trajectory(cartpole_env, cartpole_policy, max_length=10)
        for key in ["observation", "reward", "action", "terminal"]:
            assert traj[key].dtype == np.float32, (
                f"expected {key} dtype=float32, got {traj[key].dtype}"
            )

    def test_consistent_lengths(self, cartpole_env, cartpole_policy):
        traj = rollout_trajectory(cartpole_env, cartpole_policy, max_length=10)
        n = len(traj["reward"])
        for key in ["observation", "action", "next_observation", "terminal"]:
            actual = len(traj[key])
            assert actual == n, (
                f"expected len(traj['{key}'])={n} (same as reward), got {actual}"
            )


@pytest.mark.xfail(reason="TODO: get_action not yet implemented for discrete")
class TestRolloutTrajectories:
    def test_minimum_timesteps(self, cartpole_env, cartpole_policy):
        trajs, total_steps = rollout_trajectories(
            cartpole_env, cartpole_policy, min_timesteps_per_batch=50, max_length=200
        )
        assert total_steps >= 50, (
            f"expected >= 50 total steps, got {total_steps}"
        )
        assert len(trajs) >= 1, (
            f"expected >= 1 trajectory, got {len(trajs)}"
        )

    def test_returns_tuple(self, cartpole_env, cartpole_policy):
        result = rollout_trajectories(
            cartpole_env, cartpole_policy, min_timesteps_per_batch=20, max_length=10
        )
        assert isinstance(result, tuple), (
            f"expected tuple, got {type(result).__name__}"
        )
        assert len(result) == 2, (
            f"expected tuple of length 2, got {len(result)}"
        )

    def test_step_count_matches(self, cartpole_env, cartpole_policy):
        trajs, total_steps = rollout_trajectories(
            cartpole_env, cartpole_policy, min_timesteps_per_batch=30, max_length=50
        )
        actual_steps = sum(len(t["reward"]) for t in trajs)
        assert actual_steps == total_steps, (
            f"reported total_steps={total_steps} != actual sum of traj lengths={actual_steps}"
        )


@pytest.mark.xfail(reason="TODO: get_action not yet implemented for discrete")
class TestRolloutNTrajectories:
    def test_correct_count(self, cartpole_env, cartpole_policy):
        trajs = rollout_n_trajectories(
            cartpole_env, cartpole_policy, ntraj=3, max_length=10
        )
        assert len(trajs) == 3, (
            f"expected 3 trajectories, got {len(trajs)}"
        )

    def test_each_is_dict(self, cartpole_env, cartpole_policy):
        trajs = rollout_n_trajectories(
            cartpole_env, cartpole_policy, ntraj=2, max_length=10
        )
        for i, t in enumerate(trajs):
            assert isinstance(t, dict), (
                f"trajectory[{i}] expected dict, got {type(t).__name__}"
            )


class TestComputeMetrics:
    def _make_trajs(self, returns):
        return [{"reward": np.array([r], dtype=np.float32)} for r in returns]

    def test_keys(self):
        train = self._make_trajs([10.0, 20.0])
        eval_ = self._make_trajs([15.0, 25.0])
        logs = compute_metrics(train, eval_)
        for key in ["Eval_AverageReturn", "Train_AverageReturn", "Eval_StdReturn", "Train_MaxReturn"]:
            assert key in logs, (
                f"expected key '{key}' in logs, got keys: {list(logs.keys())}"
            )

    def test_eval_average(self):
        train = self._make_trajs([10.0])
        eval_ = self._make_trajs([20.0, 30.0])
        logs = compute_metrics(train, eval_)
        actual = logs["Eval_AverageReturn"]
        np.testing.assert_allclose(
            actual, 25.0,
            err_msg=f"expected Eval_AverageReturn=25.0 for returns [20, 30], got {actual}",
        )

    def test_train_max(self):
        train = self._make_trajs([5.0, 15.0, 10.0])
        eval_ = self._make_trajs([1.0])
        logs = compute_metrics(train, eval_)
        actual = logs["Train_MaxReturn"]
        assert actual == 15.0, (
            f"expected Train_MaxReturn=15.0 for returns [5, 15, 10], got {actual}"
        )

    def test_episode_lengths(self):
        train = [{"reward": np.ones(10, dtype=np.float32)}]
        eval_ = [{"reward": np.ones(5, dtype=np.float32)}]
        logs = compute_metrics(train, eval_)
        assert logs["Train_AverageEpLen"] == 10, (
            f"expected Train_AverageEpLen=10, got {logs['Train_AverageEpLen']}"
        )
        assert logs["Eval_AverageEpLen"] == 5, (
            f"expected Eval_AverageEpLen=5, got {logs['Eval_AverageEpLen']}"
        )

    def test_all_expected_keys(self):
        """compute_metrics should return all expected metric keys."""
        train = self._make_trajs([10.0, 20.0])
        eval_ = self._make_trajs([15.0, 25.0])
        logs = compute_metrics(train, eval_)
        expected_keys = {
            "Eval_AverageReturn", "Eval_StdReturn", "Eval_MaxReturn",
            "Eval_MinReturn", "Eval_AverageEpLen",
            "Train_AverageReturn", "Train_StdReturn", "Train_MaxReturn",
            "Train_MinReturn", "Train_AverageEpLen",
        }
        actual_keys = set(logs.keys())
        missing = expected_keys - actual_keys
        assert not missing, f"missing metric keys: {missing}"

    def test_eval_min_return(self):
        train = self._make_trajs([10.0])
        eval_ = self._make_trajs([5.0, 15.0, 25.0])
        logs = compute_metrics(train, eval_)
        assert logs["Eval_MinReturn"] == 5.0, (
            f"expected Eval_MinReturn=5.0, got {logs['Eval_MinReturn']}"
        )

    def test_train_std_return(self):
        train = self._make_trajs([10.0, 10.0, 10.0])
        eval_ = self._make_trajs([1.0])
        logs = compute_metrics(train, eval_)
        np.testing.assert_allclose(
            logs["Train_StdReturn"], 0.0,
            err_msg="identical returns should have std=0",
        )


class TestConvertListOfRollouts:
    def test_concatenation(self):
        trajs = [
            {
                "observation": np.ones((3, 4)),
                "action": np.ones((3, 2)),
                "next_observation": np.ones((3, 4)),
                "terminal": np.zeros(3),
                "reward": np.ones(3),
            },
            {
                "observation": np.zeros((2, 4)),
                "action": np.zeros((2, 2)),
                "next_observation": np.zeros((2, 4)),
                "terminal": np.ones(2),
                "reward": np.zeros(2),
            },
        ]
        obs, acs, next_obs, terminals, concat_rew, unconcat_rew = convert_listofrollouts(trajs)
        assert obs.shape == (5, 4), (
            f"expected obs shape (5, 4), got {obs.shape}"
        )
        assert acs.shape == (5, 2), (
            f"expected actions shape (5, 2), got {acs.shape}"
        )
        assert terminals.shape == (5,), (
            f"expected terminals shape (5,), got {terminals.shape}"
        )
        assert concat_rew.shape == (5,), (
            f"expected concatenated rewards shape (5,), got {concat_rew.shape}"
        )
        assert len(unconcat_rew) == 2, (
            f"expected 2 unconcatenated reward arrays, got {len(unconcat_rew)}"
        )


class TestGetTrajLength:
    def test_length(self):
        traj = {"reward": np.zeros(7)}
        result = get_traj_length(traj)
        assert result == 7, (
            f"expected trajectory length=7, got {result}"
        )
