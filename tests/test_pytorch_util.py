import numpy as np
import torch
from torch import nn
from cas4160.infrastructure import pytorch_util as ptu


class TestBuildMLP:
    def test_output_shape(self):
        mlp = ptu.build_mlp(input_size=4, output_size=2, n_layers=2, size=64)
        x = torch.randn(8, 4)
        out = mlp(x)
        assert out.shape == (8, 2), f"expected shape (8, 2), got {out.shape}"

    def test_zero_hidden_layers(self):
        mlp = ptu.build_mlp(input_size=4, output_size=2, n_layers=0, size=64)
        x = torch.randn(1, 4)
        out = mlp(x)
        assert out.shape == (1, 2), f"expected shape (1, 2), got {out.shape}"

    def test_single_hidden_layer(self):
        mlp = ptu.build_mlp(input_size=10, output_size=5, n_layers=1, size=32)
        x = torch.randn(4, 10)
        out = mlp(x)
        assert out.shape == (4, 5), f"expected shape (4, 5), got {out.shape}"

    def test_activation_string(self):
        mlp = ptu.build_mlp(
            input_size=4, output_size=2, n_layers=1, size=16, activation="relu"
        )
        has_relu = any(isinstance(layer, nn.ReLU) for layer in mlp)
        layer_types = [type(layer).__name__ for layer in mlp]
        assert has_relu, f"expected ReLU in layers, got: {layer_types}"

    def test_output_activation(self):
        mlp = ptu.build_mlp(
            input_size=4, output_size=2, n_layers=1, size=16, output_activation="tanh"
        )
        actual_type = type(mlp[-1]).__name__
        assert isinstance(mlp[-1], nn.Tanh), (
            f"expected last layer to be Tanh, got {actual_type}"
        )

    def test_custom_activation_module(self):
        mlp = ptu.build_mlp(
            input_size=4, output_size=2, n_layers=1, size=16, activation=nn.GELU()
        )
        has_gelu = any(isinstance(layer, nn.GELU) for layer in mlp)
        layer_types = [type(layer).__name__ for layer in mlp]
        assert has_gelu, f"expected GELU in layers, got: {layer_types}"

    def test_layer_count(self):
        n_layers = 3
        mlp = ptu.build_mlp(input_size=4, output_size=2, n_layers=n_layers, size=16)
        linear_layers = [m for m in mlp if isinstance(m, nn.Linear)]
        expected = n_layers + 1
        assert len(linear_layers) == expected, (
            f"expected {expected} Linear layers (n_layers={n_layers} hidden + 1 output), "
            f"got {len(linear_layers)}"
        )


class TestInitGPU:
    def test_cpu_fallback(self):
        ptu.init_gpu(use_gpu=False)
        assert ptu.device == torch.device("cpu"), (
            f"expected device=cpu, got {ptu.device}"
        )


class TestNumpyConversions:
    def test_from_numpy(self):
        arr = np.array([1.0, 2.0, 3.0])
        t = ptu.from_numpy(arr)
        assert isinstance(t, torch.Tensor), (
            f"expected torch.Tensor, got {type(t).__name__}"
        )
        assert t.dtype == torch.float32, (
            f"expected dtype float32, got {t.dtype}"
        )
        assert t.device == ptu.device, (
            f"expected device {ptu.device}, got {t.device}"
        )
        np.testing.assert_allclose(
            t.numpy(), arr, err_msg="from_numpy values don't match input"
        )

    def test_to_numpy(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        arr = ptu.to_numpy(t)
        assert isinstance(arr, np.ndarray), (
            f"expected np.ndarray, got {type(arr).__name__}"
        )
        np.testing.assert_allclose(
            arr, [1.0, 2.0, 3.0], err_msg="to_numpy values don't match input"
        )

    def test_roundtrip(self):
        original = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = ptu.to_numpy(ptu.from_numpy(original))
        np.testing.assert_allclose(
            result, original, err_msg="roundtrip numpy->tensor->numpy doesn't preserve values"
        )
