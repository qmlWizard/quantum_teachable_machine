from qiskit import QuantumCircuit
import numpy as np

def generate_rotation_map(rotation: str = 'ry') -> list:
    """Generate a rotation gate function based on the specified rotation type.

    Args:
        rotation (str): Type of rotation gate ('ry', 'rz', 'rxry', 'rx,rz' etc.).
    Returns:
        List of rotation gate functions.
    """

    rotation_map = []
    for char in rotation:
        if char not in ['r', 'x', 'y', 'z', ',']:
            raise ValueError(f"Unsupported rotation type: {rotation}")
        if char == 'x':
            rotation_map.append('x')
        elif char == 'y':
            rotation_map.append('y')
        elif char == 'z':
            rotation_map.append('z')

    return rotation_map


def repeat_to_qubits(x, num_qubits):
    x = np.asarray(x)

    if len(x) >= num_qubits:
        return x[:num_qubits]

    repeats = int(np.ceil(num_qubits / len(x)))
    x_repeated = np.tile(x, repeats)

    return x_repeated[:num_qubits]


def angle_encoding(x, scaling: float = 1.0, num_qubits: int = 2, rotation: str = "ry") -> QuantumCircuit:
    """Generate an embedding pattern for a given input vector.

    Args:
        x (np.ndarray): Input data vector.
        scaling (float): Scaling factor for the input data.
        num_qubits (int): Number of qubits in the quantum circuit.
        rotation (str): Type of rotation gate to use ('ry', 'rz', etc.).
    Returns:
        QuantumCircuit: A quantum circuit with the embedding pattern applied.
    """
    if num_qubits <= 0 or num_qubits < len(x)
        raise ValueError("Number of qubits must be positive and at least equal to the length of input vector x.")

    # Generate rotation map: if multiple rotations are specified, use them in sequence
    rotation_map = generate_rotation_map(rotation)

    # Create quantum circuit
    qc = QuantumCircuit(num_qubits)

    # Adjust input vector length to match number of qubits
    if num_qubits > len(x):
        x = repeat_to_qubits(x, num_qubits)

    for i in range(num_qubits):
        qubit_index = i
        angle = x[i]
        scaled_angle = angle * scaling

        for rot in rotation_map:
            if rot == 'x':
                qc.rx(scaled_angle, qubit_index)
            elif rot == 'y':
                qc.ry(scaled_angle, qubit_index)
            elif rot == 'z':
                qc.rz(scaled_angle, qubit_index)


    return qc


def bit_bit_pattern(num_qubits: int) -> QuantumCircuit:
    """Generate a bit-bit entanglement pattern for a given number of qubits.

    Args:
        num_qubits (int): Number of qubits in the quantum circuit.
    Returns:
        QuantumCircuit: A quantum circuit with the bit-bit entanglement pattern applied.
    """
    raise ValueError("Function 'bit_bit_pattern' is not implemented yet.")
