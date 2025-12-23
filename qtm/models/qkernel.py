from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit_aer.noise import NoiseModel
import numpy as np

from qtm.circuits.kernel import create_kernel_circuit


class QKernel:
    """
    Quantum Kernel class for fidelity-based kernel evaluation.

    Supports:
    - Statevector (shots=None) for deterministic kernels
    - Shot-based execution with optional noise models
    - Pluggable backends (Aer or hardware)
    """

    def __init__(
        self,
        num_qubits: int,
        encoding: str,
        ansatz: str,
        kernel_layers: int,
        embedding_rotation: str,
        scaling: float,
        reupload: bool,
        ansatz_entanglement: str,
        noise_model: NoiseModel | None = None,
        backend=None,
        shots: int | None = 1024,
    ):

        self._num_qubits = num_qubits
        self._encoding = encoding
        self._ansatz = ansatz
        self._kernel_layers = kernel_layers
        self._embedding_rotation = embedding_rotation
        self._scaling = scaling
        self._reupload = reupload
        self._ansatz_entanglement = ansatz_entanglement
        self._noise_model = noise_model
        self._backend = backend
        self._shots = shots

    # ------------------------------------------------------------------
    # Circuit construction
    # ------------------------------------------------------------------

    def construct_kernel_circuit(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        weights: np.ndarray,
    ) -> QuantumCircuit:
        """
        Build the kernel circuit |φ(x1)⟩⟨φ(x2)|.

        Assumes encoding and ansatz selection are handled internally
        by `create_kernel_circuit`.
        """
        return create_kernel_circuit(
            x1=x1,
            x2=x2,
            weights=weights,
            layers=self._kernel_layers,
            scaling=self._scaling,
            reupload=self._reupload,
            num_qubits=self._num_qubits,
            rotation=self._embedding_rotation,
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute_circuit(self, circuit):

        # Statevector path
        if self._shots is None:
            sv = Statevector.from_instruction(circuit)
            return float(np.abs(sv.data[0]) ** 2)

        # Shot-based path
        circuit = circuit.copy()
        circuit.measure_all()

        if self._backend is None:
            backend = AerSimulator(noise_model=self._noise_model) if self._noise_model else AerSimulator()
        else:
            backend = self._backend

        transpiled = transpile(circuit, backend, optimization_level=1)
        result = backend.run(transpiled, shots=self._shots).result()
        counts = result.get_counts()

        zero_state = "0" * self._num_qubits
        return counts.get(zero_state, 0) / self._shots


    # ------------------------------------------------------------------
    # Kernel API
    # ------------------------------------------------------------------

    def forward(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """
        Compute kernel value K(x1, x2).
        """

        assert weights.shape == self.parameter_shape, (
            f"Expected weights shape {self.parameter_shape}, "
            f"got {weights.shape}"
        )

        qc = self.construct_kernel_circuit(x1, x2, weights)
        return self.execute_circuit(qc)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def draw_kernel_circuit(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        weights: np.ndarray,
    ):
        """
        Draw the kernel circuit (for debugging / docs).
        """
        qc = self.construct_kernel_circuit(x1, x2, weights)
        return qc.draw("mpl")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def encoding(self) -> str:
        return self._encoding

    @property
    def ansatz(self) -> str:
        return self._ansatz

    @property
    def parameter_shape(self) -> tuple:
        """
        Shape of trainable parameters expected by the ansatz.
        """
        if self._ansatz == "embedding_pattern":
            return (self._kernel_layers, 2, self._num_qubits)

        raise NotImplementedError(
            f"Parameter shape not implemented for ansatz: {self._ansatz}"
        )
