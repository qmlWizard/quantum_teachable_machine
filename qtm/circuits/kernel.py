from typing import List, Optional, Tuple, Dict
import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
)


class AngleEmbeddingKernel:
    """
    Angle-Embedding Quantum Kernel

    Kernel Structure:
                            --- Feature Map (x1) --- Ansatz (weights) --- Ansatz (weights)† --- Feature Map (x1)† --- Measurement(Z[0])
                                -------------------------------------     -------------------------------------
                                                  |                                          |
                                          repeat for n reps                          repeat for n reps

    Kernel form:
        K(x1, x2) = | <0| U(x2)† V(θ)† V(θ) U(x1) |0> |²

    Guarantees:
    - K(x, x) = 1 (up to shot noise)
    - Symmetric
    - PSD kernel (ideal backend)
    """

    # ----------------------------
    # Initialization
    # ----------------------------
    def __init__(
        self,
        feature_dimension: int,
        num_qubits: int,
        reps: int = 1,
        entangle_inputs: Optional[List[Tuple[int, int]]] = None,
        create_superposition: bool = True,
        reupload: bool = True,
        input_scaling: bool = False,
        backend_name: str = "qasm_simulator",
        shots: int = 2048,
        noise_type: Optional[str] = None,
        noise_params: Optional[Dict] = None,
    ):
        if num_qubits < feature_dimension:
            raise ValueError("num_qubits must be >= feature_dimension")

        self.feature_dimension = feature_dimension
        self.num_qubits = num_qubits
        self.reps = reps
        self.entangle_inputs = entangle_inputs or []
        self.create_superposition = create_superposition
        self.reupload = reupload
        self.input_scaling = input_scaling

        self.backend = Aer.get_backend(backend_name)
        self.shots = shots

        self.noise_model = self._build_noise_model(noise_type, noise_params)
        self._circuit: Optional[QuantumCircuit] = None

    # ----------------------------
    # Noise models (selectable)
    # ----------------------------
    def _build_noise_model(
        self,
        noise_type: Optional[str],
        params: Optional[Dict],
    ) -> Optional[NoiseModel]:

        if noise_type is None:
            return None

        noise = NoiseModel()
        params = params or {}

        if noise_type == "depolarizing":
            p1 = params.get("p1", 0.001)
            p2 = params.get("p2", 0.01)

            noise.add_all_qubit_quantum_error(
                depolarizing_error(p1, 1),
                ["h", "ry"]
            )
            noise.add_all_qubit_quantum_error(
                depolarizing_error(p2, 2),
                ["cz", "crz"]
            )

        elif noise_type == "thermal":
            t1 = params.get("t1", 50e3)
            t2 = params.get("t2", 70e3)
            time_1q = params.get("time_1q", 50)
            time_2q = params.get("time_2q", 300)

            noise.add_all_qubit_quantum_error(
                thermal_relaxation_error(t1, t2, time_1q),
                ["h", "ry"]
            )
            noise.add_all_qubit_quantum_error(
                thermal_relaxation_error(t1, t2, time_2q),
                ["cz", "crz"]
            )

        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        return noise

    # ----------------------------
    # Utilities
    # ----------------------------
    def _repeat_to_qubits(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if len(x) >= self.num_qubits:
            return x[:self.num_qubits]
        reps = int(np.ceil(self.num_qubits / len(x)))
        return np.tile(x, reps)[:self.num_qubits]

    # ----------------------------
    # Feature map
    # ----------------------------
    def _feature_map(self, qc: QuantumCircuit, x: np.ndarray):
        for i in range(self.num_qubits):
            if self.create_superposition:
                qc.h(i)
            qc.ry(x[i], i)

        for q1, q2 in self.entangle_inputs:
            qc.cz(q1, q2)

    def _feature_map_inverse(self, qc: QuantumCircuit, x: np.ndarray):
        for q1, q2 in reversed(self.entangle_inputs):
            qc.cz(q1, q2)

        for i in range(self.num_qubits):
            qc.ry(-x[i], i)
            if self.create_superposition:
                qc.h(i)

    # ----------------------------
    # Ansatz
    # ----------------------------
    def _ansatz(self, qc: QuantumCircuit, weights):
        w_rot, w_ent = weights

        for i in range(self.num_qubits):
            qc.ry(w_rot[i], i)

        for i in range(self.num_qubits - 1):
            qc.crz(w_ent[i], i, i + 1)

    def _ansatz_inverse(self, qc: QuantumCircuit, weights):
        w_rot, w_ent = weights

        for i in reversed(range(self.num_qubits - 1)):
            qc.crz(-w_ent[i], i, i + 1)

        for i in range(self.num_qubits):
            qc.ry(-w_rot[i], i)

    # ----------------------------
    # Circuit construction
    # ----------------------------
    def _build_circuit(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        weights: List[np.ndarray],
    ) -> QuantumCircuit:

        qc = QuantumCircuit(self.num_qubits)

        x1 = self._repeat_to_qubits(x1)
        x2 = self._repeat_to_qubits(x2)

        # Forward
        self._feature_map(qc, x1)
        for r in range(self.reps):
            self._ansatz(qc, weights[r])
            if self.reupload:
                self._feature_map(qc, x1)

        self._ansatz_inverse(qc, weights[r])
        self._feature_map_inverse(qc, x2)

        # Reverse (same weights!)
        for r in reversed(range(self.reps - 1)):
            self._ansatz_inverse(qc, weights[r])
            if self.reupload:
                self._feature_map_inverse(qc, x2)

        qc.measure_all()
        self._circuit = qc
        return qc

    # ----------------------------
    # Execution
    # ----------------------------
    def execute(self, x1, x2, weights):
        qc = self._build_circuit(x1, x2, weights)

        # ----------------------------
        # Statevector (exact)
        # ----------------------------
        if self.shots is None:
            # REMOVE MEASUREMENTS
            qc_no_measure = qc.remove_final_measurements(inplace=False)
            sv = Statevector.from_instruction(qc_no_measure)
            prob_0 = np.abs(sv.data[0]) ** 2
            return prob_0
        else:
            # measurement-based execution
            qc.measure_all()
            backend = AerSimulator(noise_model=self.noise_model) if self.noise_model else AerSimulator()
            tqc = transpile(qc, backend, optimization_level=1)
            result = backend.run(tqc, shots=self.shots).result()
            counts = result.get_counts()

            zero = "0" * self.num_qubits
            return counts.get(zero, 0) / self.shots
            
    def parameter_shape(self) -> dict:
        """
        Returns the required shape of kernel parameters.

        Structure:
            weights = [
                (w_rot_0, w_ent_0),
                (w_rot_1, w_ent_1),
                ...
            ]

        where len(weights) == reps
        """

        return {
            "reps": self.reps,
            "rotation_weights_shape": (self.num_qubits,),
            "entangling_weights_shape": (self.num_qubits - 1,),
            "full_structure": [
                (
                    (self.num_qubits,),        # RY weights
                    (self.num_qubits - 1,),    # CRZ weights
                )
                for _ in range(self.reps)
            ],
        }


    # ----------------------------
    # Debug / inspection
    # ----------------------------
    def draw(self, style: str = "text"):
        if self._circuit is None:
            raise RuntimeError("Circuit not built yet.")
        return self._circuit.draw(style)

    @property
    def depth(self):
        return self._circuit.depth() if self._circuit else None

    @property
    def size(self):
        return self._circuit.size() if self._circuit else None
