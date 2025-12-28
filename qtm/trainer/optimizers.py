import numpy as np


class SPSAOptimizer:
    """
    Simultaneous Perturbation Stochastic Approximation (SPSA)

    Reference:
    Spall, J. C. (1992). Multivariate stochastic approximation
    using a simultaneous perturbation gradient approximation.
    IEEE Transactions on Automatic Control.
    """

    def __init__(
        self,
        lr: float = 0.1,
        perturbation: float = 0.1,
        a_decay: float = 0.602,
        c_decay: float = 0.101,
        seed: int | None = None,
    ):
        self.a0 = lr
        self.c0 = perturbation
        self.a_decay = a_decay
        self.c_decay = c_decay
        self.k = 0  # iteration counter

        if seed is not None:
            np.random.seed(seed)

    # --------------------------------------------------
    # Utilities (parameter-space only)
    # --------------------------------------------------

    def flatten_weights(self, weights):
        """
        weights: np.ndarray of shape (reps, P)
        returns: 1D vector
        """
        return weights.reshape(-1)

    def unflatten_weights(self, flat, reps, per_rep_dim):
        """
        flat: 1D vector
        returns: np.ndarray of shape (reps, per_rep_dim)
        """
        return flat.reshape(reps, per_rep_dim)

    # --------------------------------------------------
    # Core SPSA step
    # --------------------------------------------------

    def step(
        self,
        model,
        loss_fn,
        x0,
        x1,
        y,
        parameters,
        per_rep_dim,
        centroid=None,
    ):
        """
        Performs one SPSA update.

        Two modes:
        1) per_rep_dim != None → kernel-parameter optimization
        2) per_rep_dim == None → centroid + subcentroid optimization
        """

        # --------------------------------------------------
        # Case 1: Kernel parameter optimization
        # --------------------------------------------------
        if per_rep_dim is not None:

            theta = self.flatten_weights(parameters)
            dim = theta.shape[0]

            ak = self.a0 / ((self.k + 1) ** self.a_decay)
            ck = self.c0 / ((self.k + 1) ** self.c_decay)

            delta = np.random.choice([-1.0, 1.0], size=dim)

            theta_plus = theta + ck * delta
            theta_minus = theta - ck * delta

            params_plus = self.unflatten_weights(
                theta_plus, model.reps, per_rep_dim
            )
            params_minus = self.unflatten_weights(
                theta_minus, model.reps, per_rep_dim
            )

            K_plus = model.execute(x0, x1, params_plus)[0]
            K_minus = model.execute(x0, x1, params_minus)[0]

            loss_plus = loss_fn(K_plus, y, params_plus)
            loss_minus = loss_fn(K_minus, y, params_minus)

            grad = (loss_plus - loss_minus) / (2.0 * ck) * delta
            theta_next = theta - ak * grad

            self.k += 1

            return self.unflatten_weights(
                theta_next, model.reps, per_rep_dim
            )

        # --------------------------------------------------
        # Case 2: Centroid + subcentroid optimization
        # --------------------------------------------------
        else:
            centroid = np.asarray(centroid)
            x1 = np.asarray(x1)

            dim = centroid.shape[0]

            ak = self.a0 / ((self.k + 1) ** self.a_decay)
            ck = self.c0 / ((self.k + 1) ** self.c_decay)

            # Independent perturbations
            delta_c = np.random.choice([-1.0, 1.0], size=dim)
            delta_x = np.random.choice([-1.0, 1.0], size=dim)

            centroid_plus = centroid + ck * delta_c
            centroid_minus = centroid - ck * delta_c

            x1_plus = x1 + ck * delta_x
            x1_minus = x1 - ck * delta_x

            x0_plus = np.repeat(centroid_plus[None, :], len(x1_plus), axis=0)
            x0_minus = np.repeat(centroid_minus[None, :], len(x1_minus), axis=0)

            K_plus = model.execute(x0_plus, x1_plus, parameters)[0]
            K_minus = model.execute(x0_minus, x1_minus, parameters)[0]

            loss_plus = loss_fn(K_plus, y, centroid_plus)
            loss_minus = loss_fn(K_minus, y, centroid_minus)

            grad_c = (loss_plus - loss_minus) / (2.0 * ck) * delta_c
            grad_x = (loss_plus - loss_minus) / (2.0 * ck) * delta_x

            centroid_next = centroid - ak * grad_c
            x1_next = x1 - ak * grad_x

            self.k += 1

            return centroid_next, x1_next
