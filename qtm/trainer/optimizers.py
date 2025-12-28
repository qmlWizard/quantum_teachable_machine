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
        """
        Parameters
        ----------
        lr : float
            Base learning rate (a_0)

        perturbation : float
            Base perturbation magnitude (c_0)

        a_decay : float
            Learning rate decay exponent (0.5 < a_decay ≤ 1)

        c_decay : float
            Perturbation decay exponent (≈ 0.1 typical)

        seed : int or None
            Random seed for reproducibility
        """

        self.a0 = lr
        self.c0 = perturbation
        self.a_decay = a_decay
        self.c_decay = c_decay

        self.k = 0  # iteration counter

        if seed is not None:
            np.random.seed(seed)

    # --------------------------------------------------
    # Core SPSA step
    # --------------------------------------------------

    def step(self, loss_fn, parameters):
        """
        Perform one SPSA update step.

        Parameters
        ----------
        loss_fn : callable
            Function that accepts a parameter vector and returns scalar loss.
            Example:
                loss_fn(theta) -> float

        parameters : np.ndarray
            Current parameter vector (θ)

        Returns
        -------
        np.ndarray
            Updated parameter vector
        """

        theta = parameters.copy()
        dim = theta.shape[0]

        # --- Step schedules ---
        ak = self.a0 / ((self.k + 1) ** self.a_decay)
        ck = self.c0 / ((self.k + 1) ** self.c_decay)

        # --- Random perturbation vector (±1) ---
        delta = np.random.choice([-1.0, 1.0], size=dim)

        # --- Perturbed parameters ---
        theta_plus = theta + ck * delta
        theta_minus = theta - ck * delta

        # --- Two loss evaluations ---
        loss_plus = loss_fn(theta_plus)
        loss_minus = loss_fn(theta_minus)

        # --- SPSA gradient estimate ---
        grad_estimate = (loss_plus - loss_minus) / (2.0 * ck) * delta

        # --- Parameter update ---
        theta_next = theta - ak * grad_estimate

        # Increment iteration counter
        self.k += 1

        return theta_next
