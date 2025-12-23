from qiskit import QuantumCircuit
import numpy as np


def generate_rotation_map(rotation: str = "ry") -> list:
    """
    Generate a rotation gate map based on the specified rotation type.

    Args:
        rotation (str): Type of rotation gate ('ry', 'rz', 'rxry', etc.).

    Returns:
        list: List of rotation identifiers (e.g. ['r', 'y']).
    """
    rotation_map = []

    for char in rotation:
        if char not in ["r", "x", "y", "z", ","]:
            raise ValueError(f"Unsupported rotation type: {rotation}")

        if char == "x":
            rotation_map.append("x")
        elif char == "y":
            rotation_map.append("y")
        elif char == "z":
            rotation_map.append("z")

    return rotation_map


def embedding_pattern(weights: np.ndarray, num_qubits: int = 2) -> QuantumCircuit:
    """
    Generate an embedding pattern ansatz.

    Args:
        weights (np.ndarray): Trainable parameters of shape (2, num_qubits).
        num_qubits (int): Number of qubits in the quantum circuit.

    Returns:
        QuantumCircuit: A quantum circuit with the embedding pattern applied.
    """

    if num_qubits <= 0:
        raise ValueError("Number of qubits must be positive.")

    qc = QuantumCircuit(num_qubits)

    weights_straight = weights[0]
    weights_entangle = weights[1]

    # Single-qubit rotations
    for i in range(num_qubits):
        qc.ry(weights_straight[i], i)

    # Entangling layer (linear chain)
    for i in range(num_qubits - 1):
        qc.crz(weights_entangle[i], i, i + 1)

    return qc
