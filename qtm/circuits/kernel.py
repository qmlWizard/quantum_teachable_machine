from qiskit import QuantumCircuit
import numpy as np

from qtm.circuits.feature_maps import angle_encoding
from qtm.circuits.ansatz import embedding_pattern


class AngleEmbeddingKernel:

    """
        Kernel Structure:
                            --- Feature Map (x1) --- Ansatz (weights) --- Ansatz (weights)† --- Feature Map (x1)† --- Measurement(Z[0])
                                -------------------------------------     -------------------------------------
                                                  |                                          |
                                          repeat for n reps                          repeat for n reps
        Parameters:
    """

    def __init__(self, 
                 feature_dimension: int, 
                 reps: int = 1,
                 num_qubits: int = 0,
                 trainability: bool = True,
                 input_scaling: bool = True,
                 reupload: bool = True,
                 entangle_inputs: List = [],
                 create_superposition: bool = True,
                ):

        self._feature_dimension = feature_dimension
        self._reps = reps
        self._num_qubits = num_qubits
        self._trainable = trainability
        self._scaling = input_scaling
        self._reupload = reupload
        self._featuremap_entangle = entangle_inputs
        self._explode = create_superposition

        if self._num_qubits <= 0 or self._num_qubits < self._feature_dimension:
            raise ValueError("Number of qubits must be positive and at least equal to the length of input vector x.")

        self._circuit = QuantumCircuit(self._num_qubits)

    def repeat_to_qubits(self, x):
        x = np.asarray(x)

        if len(x) >= self._num_qubits:
            return x[:self._num_qubits]

        repeats = int(np.ceil(self._num_qubits / len(x)))
        x_repeated = np.tile(x, repeats)

        return x_repeated[:num_qubits]

    def build_feature_map(self, x: np.ndarray) -> QuantumCircuit:

        for i in range(self._num_qubits):
            qubit_index = i
            angle = x[i]
            self._circuit.h(qubit_index) if self._explode else 'Damn !! No Dimentionality Explosion'
            self._circuit.ry(scaled_angle, qubit_index)
        
        if len(self._featuremap_entangle) > 0 :
            for (q1, q2) in self._featuremap_entangle:
                self._circuit.cz(q1, q2)

    def build_feature_map_reverse(self, x: np.ndarray, scaling) -> QuantumCircuit:
        if len(self._featuremap_entangle) > 0 :
            for (q1, q2) in self._featuremap_entangle:
                self._circuit.cz(q1, q2)

        for i in range(self._num_qubits):
            qubit_index = i
            angle = x[i]
            self._circuit.ry(scaled_angle, qubit_index)
            self._circuit.h(qubit_index) if self._explode else 'Damn !! No Dimentionality Explosion'
            
        

    def build_ansatz(self, weights: np.ndarray) -> QuantumCircuit:
        pass

    def build_ansatz_reverse(self, weights: np.ndarray) -> QuantumCircuit:
        pass

    def _build_circuit(self, x1, x2, weights):
        x1 = repeat_to_qubits(x1)
        x2 = repeat_to_qubits(x2)

        if self._scaling:
            x1 = x1 * weights[0][1]
            x2 = x2 * weights[0][2]

        self.build_feature_map(x1)

        for rep in self._reps:
            self._build_ansatz(weights[1][rep])
            self._build_feature_map(x1) if reupload else "Reuploading not selected continuing with next process"

        self.build_ansatz_reverse(weights[2][rep])
        self.build_feature_map_reverse(x2)

        for rep in self._reps - 1:
            self.build_ansatz_reverse(weights[2][rep])
            self.build_feature_map_reverse(x2) if reupload else "Reuploading not selected continuing with next process"

        self._circuit.measure_all()


    def kernel_function(self, x1, x2, weights):
        return self._build_circuit(x1, x2, weights)