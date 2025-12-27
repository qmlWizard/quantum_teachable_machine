from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit_aer.noise import NoiseModel
import numpy as np
from qtm.circuits.kernel import AngleEmbeddingKernel

class QKernel:
    def __init__(self, feature_dimension, num_qubits, reps=1, shots=2048):
        self.kernel = AngleEmbeddingKernel(
            feature_dimension=feature_dimension,
            num_qubits=num_qubits,
            reps=reps,
            shots=shots
        )

    def fit(self, training_data, training_labels):
        pass

    def predict(self, data):
        pass

    