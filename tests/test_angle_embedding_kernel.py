import numpy as np
import pytest
from qtm.circuits.kernel import AngleEmbeddingKernel


def unit_weights(kernel):
    return [
        (
            np.ones(kernel.num_qubits),
            np.ones(kernel.num_qubits - 1),
        )
        for _ in range(kernel.reps)
    ]


def random_weights(kernel, seed=0):
    rng = np.random.default_rng(seed)
    return [
        (
            rng.standard_normal(kernel.num_qubits),
            rng.standard_normal(kernel.num_qubits - 1),
        )
        for _ in range(kernel.reps)
    ]


def sample_x(dim):
    return np.linspace(0.1, 0.1 * dim, dim)


# ------------------------------------------------------------------
# Basic execution tests (valid for current kernel)
# ------------------------------------------------------------------

def test_execute_returns_float():
    kernel = AngleEmbeddingKernel(3, 3, reps=1, shots=1024)
    x = sample_x(3)
    w = unit_weights(kernel)

    val = kernel.execute(x, x, w)
    assert isinstance(val, float)


def test_kernel_value_in_range():
    kernel = AngleEmbeddingKernel(3, 3, reps=1, shots=1024)
    x = np.array([0.1, 0.2, 0.3])
    y = np.array([0.3, 0.2, 0.1])
    w = random_weights(kernel)

    val = kernel.execute(x, y, w)
    assert 0.0 <= val <= 1.0


def test_kernel_not_nan():
    kernel = AngleEmbeddingKernel(3, 3, reps=2, shots=1024)
    x = sample_x(3)
    w = random_weights(kernel)

    val = kernel.execute(x, x, w)
    assert np.isfinite(val)


def test_shot_variance_reasonable():
    kernel = AngleEmbeddingKernel(3, 3, reps=1, shots=2048)
    x = sample_x(3)
    w = unit_weights(kernel)

    values = [kernel.execute(x, x, w) for _ in range(5)]
    assert np.std(values) < 0.1


def test_invalid_weight_shape_raises():
    kernel = AngleEmbeddingKernel(3, 3, reps=1, shots=1024)
    x = sample_x(3)

    bad_weights = [(np.ones(2), np.ones(2))]
    with pytest.raises(Exception):
        kernel.execute(x, x, bad_weights)


def test_multiple_calls_do_not_crash():
    kernel = AngleEmbeddingKernel(3, 3, reps=1, shots=1024)
    x = sample_x(3)
    w = unit_weights(kernel)

    for _ in range(3):
        val = kernel.execute(x, x, w)
        assert isinstance(val, float)
