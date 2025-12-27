"""
Flow
    Get Centroids
        → Optimizer (SPSA / Parameter Shift)
            → CCKA Loss
                → Quantum Kernel Circuit
                    → Update Centroids
"""

import time
import numpy as np
from qiskit import QuantumCircuit

from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score
)


class CentroidKernelTrainer:

    def __init__(
        self,
        model: QuantumCircuit,
        optimizer: str = "spsa",
        outer_steps: int = 5,
        inner_steps: int = 5,
        num_subcentroids: int = 2,
        lr_param: float = 0.1,
        lr_centroid: float = 0.5,
        lr_subcentroid: float = 0.5,
        lambda_kao: float = 0.01,
        lambda_co: float = 0.01,
        X: np.ndarray = None,
        y: np.ndarray = None,
        training_split: float = 0.8,
        validation_split: float = 0.1,
    ):

        # ----------------------------
        # Validation
        # ----------------------------
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

        self._model = model
        self._optimizer = optimizer
        self._outer_steps = outer_steps
        self._inner_steps = inner_steps
        self._num_subcentroids = num_subcentroids
        self._lr_param = lr_param
        self._lr_centroid = lr_centroid
        self._lr_subcentroid = lr_subcentroid
        self._lambda_kao = lambda_kao
        self._lambda_co = lambda_co
        self._validation_split = validation_split

        # ----------------------------
        # Classes 
        # ----------------------------
        self._classes = np.unique(y)
        self._num_classes = len(self._classes)

        # ----------------------------
        # Centroids
        # ----------------------------
        feature_dim = X.shape[1]

        self._centroids = np.zeros((self._num_classes, feature_dim))
        self._subcentroids = np.zeros(
            (self._num_classes * num_subcentroids, feature_dim)
        )

        self._centroids_cls = np.zeros(self._num_classes, dtype=int)
        self._subcentroids_cls = np.zeros(
            self._num_classes * num_subcentroids, dtype=int
        )

        # ----------------------------
        # Optimizers
        # ----------------------------
        if optimizer == "spsa":
            from qtm.trainer.optimizers import SPSAOptimizer
            self._param_optm = SPSAOptimizer(lr=self._lr_param)
            self._centroid_optm = SPSAOptimizer(lr=self._lr_centroid)
            self._subcentroid_optm = SPSAOptimizer(lr=self._lr_subcentroid)
        elif optimizer == "param_shift":
            from qtm.trainer.optimizers import ParameterShiftOptimizer
            self._param_optm = ParameterShiftOptimizer(lr=self._lr_param)
            self._centroid_optm = ParameterShiftOptimizer(lr=self._lr_centroid)
            self._subcentroid_optm = ParameterShiftOptimizer(lr=self._lr_subcentroid)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        self._weights = np.random.randn(self._model.parameter_shape())

        # ----------------------------
        # Data split
        # ----------------------------
        n = X.shape[0]
        train_end = int(n * training_split)
        val_size = int(train_end * validation_split)

        self.X_train = X[:train_end - val_size]
        self.y_train = y[:train_end - val_size]
        self.X_val = X[train_end - val_size:train_end]
        self.y_val = y[train_end - val_size:train_end]
        self.X_test = X[train_end:]
        self.y_test = y[train_end:]

        self._init_centroids(self.X_train, self.y_train)

    def _init_centroids(self, X, y):
        for idx, cls in enumerate(self._classes):
            class_data = X[y == cls]

            self._centroids[idx] = class_data.mean(axis=0)
            self._centroids_cls[idx] = cls

            splits = np.array_split(class_data, self._num_subcentroids)
            for s_idx, split in enumerate(splits):
                flat_idx = idx * self._num_subcentroids + s_idx
                self._subcentroids[flat_idx] = split.mean(axis=0)
                self._subcentroids_cls[flat_idx] = cls

    def _centroid_kta(self, K, y):
        K = K.reshape(-1)
        y = y.reshape(-1)

        if K.shape != y.shape:
            raise ValueError("Kernel and labels must have same shape")

        num = np.dot(K, y)
        den = np.linalg.norm(K) * np.linalg.norm(y)
        return num / den

    def _kta(self, K, y):
        y = y.reshape(-1, 1)
        T = y @ y.T
        return np.sum(K * T) / (
            np.linalg.norm(K, "fro") * np.linalg.norm(T, "fro")
        )

    def _loss_kao(self, K, y, weights):
        kta = self._centroid_kta(K, y)
        reg = self._lambda_kao * np.sum(weights ** 2)
        return 1.0 - kta + reg

    def _loss_co(self, K, y, centroid):
        kta = self._centroid_kta(K, y)
        reg = np.sum(
            np.maximum(centroid - 1.0, 0.0) +
            np.maximum(-centroid, 0.0)
        )
        return 1.0 - kta + self._lambda_co * reg

    def _psd(self, K, tol=1e-8):
        eigvals = np.linalg.eigvalsh(K)
        return {
            "min_eigenvalue": float(eigvals.min()),
            "num_negative_eigenvalues": int(np.sum(eigvals < -tol)),
            "psd_violation_magnitude": float(np.sum(np.abs(eigvals[eigvals < 0])))
        }

    def _kcn(self, K, eps=1e-12):
        eigvals = np.linalg.eigvalsh(K)
        return float(eigvals.max() / max(eigvals.min(), eps))

    def _krm(self, K, tol=1e-10):
        eigvals = np.linalg.eigvalsh(K)
        rank = np.sum(eigvals > tol)
        p = eigvals / np.sum(eigvals)
        p = p[p > 0]
        eff_rank = np.exp(-np.sum(p * np.log(p)))
        return {
            "rank": int(rank),
            "effective_rank": float(eff_rank),
        }

    def _svm_margin(self, dual_coef, support_labels, support_kernel):
        alpha_y = dual_coef.reshape(-1, 1)
        Q = support_kernel * (
            support_labels[:, None] * support_labels[None, :]
        )
        w_norm_sq = alpha_y.T @ Q @ alpha_y
        return float(1.0 / np.sqrt(w_norm_sq))

    def train(self):
        start = time.time()

        for _ in range(self._outer_steps):
            for c_idx, cls in enumerate(self._classes):

                start_idx = c_idx * self._num_subcentroids
                end_idx = start_idx + self._num_subcentroids

                centroid = self._centroids[c_idx]
                subc = self._subcentroids[start_idx:end_idx]
                labels = np.where(
                    self._subcentroids_cls[start_idx:end_idx] == cls,
                    1, -1
                )

                # ---- parameter optimization ----
                for _ in range(self._inner_steps):
                    x0 = np.repeat(centroid[None, :], len(subc), axis=0)
                    K = self._model.execute(x0, subc, self._weights)

                    loss = self._loss_kao(K, labels, self._weights)
                    self._weights = self._param_optm.step(
                        loss, self._model.parameters
                    )

                # ---- centroid optimization ----
                for _ in range(self._inner_steps):
                    x0 = np.repeat(centroid[None, :], len(subc), axis=0)
                    K = self._model.execute(x0, subc, self._weights)

                    loss = self._loss_co(K, labels, centroid)
                    centroid, subc = self._centroid_optm.step(
                        loss, centroid, subc
                    )

                    self._centroids[c_idx] = centroid
                    self._subcentroids[start_idx:end_idx] = subc

        self.training_time = time.time() - start

    def evaluate(self):

        evaluation = {}

        # ---- Train kernel ----
        Xtr = self.X_train
        K_train = self._model.execute(
            np.repeat(Xtr, len(Xtr), axis=0),
            np.tile(Xtr, (len(Xtr), 1)),
            self._weights
        )

        evaluation["kernel_alignment"] = self._kta(K_train, self.y_train)
        evaluation.update(self._psd(K_train))
        evaluation["condition_number"] = self._kcn(K_train)
        evaluation.update(self._krm(K_train))

        clf = SVC(kernel="precomputed", max_iter=10000)
        clf.fit(K_train, self.y_train)

        # ---- Test kernel ----
        K_test = self._model.execute(
            np.repeat(self.X_test, len(self.X_train), axis=0),
            np.tile(self.X_train, (len(self.X_test), 1)),
            self._weights
        )

        preds = clf.predict(K_test)

        evaluation["accuracy"] = accuracy_score(self.y_test, preds)
        evaluation["precision"] = precision_score(
            self.y_test, preds, average="macro"
        )
        evaluation["recall"] = recall_score(
            self.y_test, preds, average="macro"
        )
        evaluation["f1"] = f1_score(
            self.y_test, preds, average="macro"
        )

        # ---- SVM geometry ----
        sv = clf.support_
        support_kernel = K_train[np.ix_(sv, sv)]
        evaluation["svm_margin"] = self._svm_margin(
            clf.dual_coef_, clf._y[sv], support_kernel
        )
        evaluation["num_support_vectors"] = len(sv)

        return evaluation
