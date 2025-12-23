import numpy as np
import pytest

from qtm.models.qkernel import QKernel


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def basic_kernel():
    """
    Deterministic statevector kernel for testing.
    """
    return QKernel(
        num_qubits=3,
        encoding="angle",
        ansatz="embedding_pattern",
        kernel_layers=1,
        embedding_rotation="ry",
        scaling=1.0,
        reupload=False,
        ansatz_entanglement="linear",
        shots=None,  # statevector mode
    )


@pytest.fixture
def random_inputs():
    x1 = np.array([0.1, 0.5, 0.9])
    x2 = np.array([0.2, 0.4, 0.8])
    return x1, x2


@pytest.fixture
def weights(basic_kernel):
    return np.random.uniform(
    0, 2*np.pi,
    size=(basic_kernel._kernel_layers, 2, basic_kernel._num_qubits)
)

# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_kernel_output_range(basic_kernel, random_inputs, weights):
    """
    Kernel values must lie in [0, 1].
    """
    x1, x2 = random_inputs
    k = basic_kernel.forward(x1, x2, weights)

    assert isinstance(k, float)
    assert 0.0 <= k <= 1.0


def test_kernel_self_similarity(basic_kernel, random_inputs, weights):
    """
    K(x, x) should be ~1 in statevector mode.
    """
    x, _ = random_inputs
    k_xx = basic_kernel.forward(x, x, weights)

    assert np.isclose(k_xx, 1.0, atol=1e-6)


def test_kernel_symmetry(basic_kernel, random_inputs, weights):
    """
    K(x1, x2) == K(x2, x1)
    """
    x1, x2 = random_inputs

    k12 = basic_kernel.forward(x1, x2, weights)
    k21 = basic_kernel.forward(x2, x1, weights)

    assert np.isclose(k12, k21, atol=1e-6)


def test_parameter_shape_validation(basic_kernel, random_inputs):
    """
    Invalid weight shapes must raise an error.
    """
    x1, x2 = random_inputs
    bad_weights = np.random.rand(3, 3)

    with pytest.raises(AssertionError):
        basic_kernel.forward(x1, x2, bad_weights)


def test_circuit_construction(basic_kernel, random_inputs, weights):
    """
    Circuit must be constructible and non-empty.
    """
    x1, x2 = random_inputs
    qc = basic_kernel.construct_kernel_circuit(x1, x2, weights)

    assert qc is not None
    assert qc.num_qubits == basic_kernel._num_qubits
    assert len(qc.data) > 0


def test_shot_based_execution(random_inputs, weights):
    """
    Shot-based execution should run without crashing.
    """
    kernel = QKernel(
        num_qubits=3,
        encoding="angle",
        ansatz="embedding_pattern",
        kernel_layers=1,
        embedding_rotation="ry",
        scaling=1.0,
        reupload=False,
        ansatz_entanglement="linear",
        shots=256,
    )

    x1, x2 = random_inputs
    k = kernel.forward(x1, x2, weights)

    assert isinstance(k, float)
    assert 0.0 <= k <= 1.0
