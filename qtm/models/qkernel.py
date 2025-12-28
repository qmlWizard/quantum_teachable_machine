from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class QKernel:
    def __init__(
        self,
        kernel_circuit: QuantumCircuit,
        matrix_type: str = "regular",
        matrix_normalisation: bool = True,
        landmark_points: int = 0,
        parallelise: bool = True,
        visualise_tqdm: bool = False,
    ):
        self._kernel_circuit = kernel_circuit
        self._matrix_type = matrix_type
        self._matrix_normalisation = matrix_normalisation
        self._landmark_points = landmark_points
        self._parallelise = parallelise
        self._visualise_tqdm = visualise_tqdm

        self._max_workers = self.auto_max_workers(mode="auto", cap=10)
        self.num_qubits = kernel_circuit.num_qubits
        self.reps = kernel_circuit.reps

    # ==========================================================
    # Utilities
    # ==========================================================

    def auto_max_workers(self, mode="auto", cap=10):
        cpu = os.cpu_count() or 1
        if mode == "io":
            return min(4 * cpu, cap)
        elif mode == "cpu":
            return cpu
        elif mode == "mixed":
            return min(4 * cpu, cap)
        return min(cpu, cap)

    def _progress(self, iterable=None, **kwargs):
        """
        tqdm wrapper that becomes a no-op if disabled.
        """
        if self._visualise_tqdm and tqdm is not None:
            return tqdm(iterable, **kwargs)
        return iterable

    # ==========================================================
    # Weights handling (UNCHANGED)
    # ==========================================================

    def _unpack_weights(self, weights):
        if isinstance(weights, list):
            return weights

        structured = []
        rot_dim = self.num_qubits
        ent_dim = self.num_qubits - 1

        for r in range(self._kernel_circuit.reps):
            row = weights[r]
            w_rot = row[:rot_dim]
            w_ent = row[rot_dim:rot_dim + ent_dim]
            structured.append((w_rot, w_ent))

        return structured

    # ==========================================================
    # SERIAL EXECUTION
    # ==========================================================

    def execute_serial(self, x0, x1, weights):
        n0, n1 = len(x0), len(x1)
        K = np.zeros((n0, n1))

        if x0 is x1 or (n0 == n1 and np.array_equal(x0, x1)):
            iu, ju = np.triu_indices(n0, k=1)

            for i, j in self._progress(
                zip(iu, ju),
                total=len(iu),
                desc="Kernel Matrix (serial, triangular)",
            ):
                val = self._kernel_circuit.execute(x0[i], x0[j], weights)
                K[i, j] = val
                K[j, i] = val

            np.fill_diagonal(K, 1.0)
            return K

        for i in self._progress(
            range(n0),
            desc="Kernel Matrix (serial, full)",
        ):
            for j in range(n1):
                K[i, j] = self._kernel_circuit.execute(x0[i], x1[j], weights)

        return K

    # ==========================================================
    # PARALLEL EXECUTION
    # ==========================================================

    def _kernel_worker(self, args):
        i, j, x_i, x_j, weights, kernel_factory = args
        if x_i is x_j:
            return i, j, 1.0
        kernel = kernel_factory()
        val = kernel.execute(x_i, x_j, weights)
        return i, j, val

    def _kernel_factory(self):
        return self._kernel_circuit

    def execute_parallel(self, x0, x1, weights):
        n0, n1 = len(x0), len(x1)
        K = np.zeros((n0, n1))
        ctx = mp.get_context("spawn")

        if x0 is x1 or (n0 == n1 and np.array_equal(x0, x1)):
            iu, ju = np.triu_indices(n0, k=1)
            tasks = [
                (i, j, x0[i], x0[j], weights, self._kernel_factory)
                for i, j in zip(iu, ju)
            ]

            with ProcessPoolExecutor(
                max_workers=self._max_workers,
                mp_context=ctx,
            ) as executor:
                futures = [
                    executor.submit(self._kernel_worker, task)
                    for task in tasks
                ]

                for fut in self._progress(
                    as_completed(futures),
                    total=len(futures),
                    desc="Kernel Matrix (process, triangular)",
                ):
                    i, j, val = fut.result()
                    K[i, j] = val
                    K[j, i] = val

            np.fill_diagonal(K, 1.0)
            return K

        tasks = [
            (i, j, x0[i], x1[j], weights, self._kernel_factory)
            for i in range(n0)
            for j in range(n1)
        ]

        with ProcessPoolExecutor(
            max_workers=self._max_workers,
            mp_context=ctx,
        ) as executor:
            futures = [
                executor.submit(self._kernel_worker, task)
                for task in tasks
            ]

            for fut in self._progress(
                as_completed(futures),
                total=len(futures),
                desc="Kernel Matrix (process, full)",
            ):
                i, j, val = fut.result()
                K[i, j] = val

        return K

    # ==========================================================
    # NYSTRÖM
    # ==========================================================

    def nystrom_kernel(self, x0, x1, weights):
        if self._landmark_points <= 0:
            raise ValueError("landmark_points must be > 0")

        landmarks = x1[:self._landmark_points]
        K_nm = self.execute_parallel(x0, landmarks, weights)
        K_mm = self.execute_parallel(landmarks, landmarks, weights)

        reg = 1e-8 * np.eye(len(landmarks))
        return K_nm @ np.linalg.inv(K_mm + reg) @ K_nm.T

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def regular_kernel(self, x0, x1, weights):
        if self._parallelise:
            return self.execute_parallel(x0, x1, weights)
        return self.execute_serial(x0, x1, weights)

    def execute(self, x0, x1, weights):
        weights = self._unpack_weights(weights)
        self._state_cache = {}

        if self._matrix_type == "nystrom":
            return self.nystrom_kernel(x0, x1, weights)

        return self.regular_kernel(x0, x1, weights)
