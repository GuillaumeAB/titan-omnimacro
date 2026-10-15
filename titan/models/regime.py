"""
titan.models.regime — Gaussian Hidden Markov Model (HMM) for market regimes.
Pure numpy/scipy Baum-Welch implementation in log-space with multiple restarts,
vectorized emission densities, forward filter, smoother, Viterbi decoding,
and automatic state ordering (State 0 = Calmest, State K-1 = Most Stressed).
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


class GaussianHMM:
    """
    Gaussian Hidden Markov Model with full covariance matrices.
    Fitted via Expectation-Maximization (Baum-Welch) in numerically stable log-space.
    """

    def __init__(
        self,
        n_states: int = 3,
        n_restarts: int = 4,
        max_iter: int = 150,
        tol: float = 1e-5,
        seed: int = 42,
        cov_floor: float = 1e-6,
    ):
        self.n_states = n_states
        self.n_restarts = n_restarts
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.cov_floor = cov_floor

        self.startprob_: np.ndarray | None = None
        self.transmat_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.covars_: np.ndarray | None = None
        self.loglik_: float = -np.inf
        self.state_order_: np.ndarray | None = None

    @property
    def state_names(self) -> list[str]:
        if self.n_states == 2:
            return ["Calm", "Stress"]
        if self.n_states == 3:
            return ["Calm", "Normal", "Stress"]
        return [f"Regime {i}" for i in range(self.n_states)]

    def _init_params(self, x: np.ndarray, rng: np.random.Generator):
        t, d = x.shape
        k = self.n_states

        means = np.zeros((k, d))
        idx = rng.choice(t)
        means[0] = x[idx]

        for step in range(1, k):
            dists = np.min([np.sum((x - means[j]) ** 2, axis=1) for j in range(step)], axis=0)
            probs = dists / np.sum(dists)
            idx = rng.choice(t, p=probs)
            means[step] = x[idx]

        base_cov = np.cov(x, rowvar=False)
        if d == 1:
            base_cov = np.array([[float(base_cov)]])
        covars = np.zeros((k, d, d))
        for step in range(k):
            covars[step] = base_cov + np.eye(d) * self.cov_floor

        trans = np.full((k, k), 0.05 / (k - 1) if k > 1 else 1.0)
        np.fill_diagonal(trans, 0.95)
        startprob = np.ones(k) / k

        return startprob, trans, means, covars

    def _compute_log_emission(self, x: np.ndarray, means: np.ndarray, covars: np.ndarray) -> np.ndarray:
        t, d = x.shape
        k = self.n_states
        log_b = np.zeros((t, k))
        const = -0.5 * d * np.log(2.0 * np.pi)

        for step in range(k):
            diff = x - means[step]
            cov_k = covars[step] + np.eye(d) * self.cov_floor
            try:
                sign, logdet = np.linalg.slogdet(cov_k)
                if sign <= 0:
                    cov_k += np.eye(d) * 1e-4
                    sign, logdet = np.linalg.slogdet(cov_k)
                inv_cov = np.linalg.pinv(cov_k)
                quad = np.sum((diff @ inv_cov) * diff, axis=1)
                log_b[:, step] = const - 0.5 * logdet - 0.5 * quad
            except Exception:
                log_b[:, step] = -1e10

        return log_b

    def _forward_backward(self, log_b: np.ndarray, startprob: np.ndarray, transmat: np.ndarray):
        t, k = log_b.shape
        log_pi = np.log(np.maximum(startprob, 1e-15))
        log_a = np.log(np.maximum(transmat, 1e-15))

        log_alpha = np.zeros((t, k))
        log_alpha[0] = log_pi + log_b[0]
        for step in range(1, t):
            for j in range(k):
                log_alpha[step, j] = logsumexp(log_alpha[step - 1] + log_a[:, j]) + log_b[step, j]

        loglik = logsumexp(log_alpha[-1])

        log_beta = np.zeros((t, k))
        log_beta[-1] = 0.0
        for step in range(t - 2, -1, -1):
            for i in range(k):
                log_beta[step, i] = logsumexp(log_a[i, :] + log_b[step + 1, :] + log_beta[step + 1, :])

        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)

        log_xi = np.zeros((t - 1, k, k))
        for step in range(t - 1):
            term = log_alpha[step, :, None] + log_a + log_b[step + 1, None, :] + log_beta[step + 1, None, :]
            log_xi[step] = term - logsumexp(term)
        xi = np.exp(log_xi)

        return loglik, log_alpha, gamma, xi

    def fit(self, x: np.ndarray) -> GaussianHMM:
        """Fit Gaussian HMM with multiple random restarts."""
        x = np.asarray(x, dtype=float)
        _, d = x.shape
        k = self.n_states

        best_ll = -np.inf
        best_params = None

        master_rng = np.random.default_rng(self.seed)

        for _ in range(self.n_restarts):
            rng = np.random.default_rng(master_rng.integers(100000))
            pi, a, mu, sigma = self._init_params(x, rng)
            prev_ll = -np.inf

            for _ in range(self.max_iter):
                log_b = self._compute_log_emission(x, mu, sigma)
                ll, _, gamma, xi = self._forward_backward(log_b, pi, a)

                if np.isnan(ll):
                    break
                if abs(ll - prev_ll) < self.tol:
                    break
                prev_ll = ll

                pi = gamma[0] / np.sum(gamma[0])
                a = np.sum(xi, axis=0) / np.maximum(np.sum(gamma[:-1], axis=0)[:, None], 1e-12)
                a = a / np.sum(a, axis=1, keepdims=True)

                gamma_sum = np.maximum(np.sum(gamma, axis=0), 1e-12)
                mu = (gamma.T @ x) / gamma_sum[:, None]

                for step in range(k):
                    diff = x - mu[step]
                    sigma[step] = (diff.T @ (diff * gamma[:, step, None])) / gamma_sum[step]
                    sigma[step] += np.eye(d) * self.cov_floor

            if prev_ll > best_ll:
                best_ll = prev_ll
                best_params = (pi, a, mu, sigma)

        if best_params is None:
            raise RuntimeError("Gaussian HMM failed to converge.")

        pi, a, mu, sigma = best_params

        # Re-order states by ascending variance of first factor (MKT)
        vols = np.array([sigma[step, 0, 0] for step in range(k)])
        order = np.argsort(vols)
        self.state_order_ = order

        self.startprob_ = pi[order]
        self.transmat_ = a[order][:, order]
        self.means_ = mu[order]
        self.covars_ = sigma[order]
        self.loglik_ = best_ll
        return self

    def filter(self, x: np.ndarray) -> np.ndarray:
        """Forward filtered probabilities P(s_t | x_1..t). Shape (T, K)."""
        x = np.asarray(x, dtype=float)
        log_b = self._compute_log_emission(x, self.means_, self.covars_)
        t, k = log_b.shape
        log_pi = np.log(np.maximum(self.startprob_, 1e-15))
        log_a = np.log(np.maximum(self.transmat_, 1e-15))

        log_alpha = np.zeros((t, k))
        log_alpha[0] = log_pi + log_b[0]
        for step in range(1, t):
            for j in range(k):
                log_alpha[step, j] = logsumexp(log_alpha[step - 1] + log_a[:, j]) + log_b[step, j]

        norm = log_alpha - logsumexp(log_alpha, axis=1, keepdims=True)
        return np.exp(norm)

    def smooth(self, x: np.ndarray) -> np.ndarray:
        """Smoothed state probabilities P(s_t | x_1..T). Shape (T, K)."""
        x = np.asarray(x, dtype=float)
        log_b = self._compute_log_emission(x, self.means_, self.covars_)
        _, _, gamma, _ = self._forward_backward(log_b, self.startprob_, self.transmat_)
        return gamma

    def viterbi(self, x: np.ndarray) -> np.ndarray:
        """Most likely state sequence (Viterbi path). Shape (T,)."""
        x = np.asarray(x, dtype=float)
        log_b = self._compute_log_emission(x, self.means_, self.covars_)
        t, k = log_b.shape
        log_pi = np.log(np.maximum(self.startprob_, 1e-15))
        log_a = np.log(np.maximum(self.transmat_, 1e-15))

        viterbi_log = np.zeros((t, k))
        backpointer = np.zeros((t, k), dtype=int)

        viterbi_log[0] = log_pi + log_b[0]
        for step in range(1, t):
            for j in range(k):
                vals = viterbi_log[step - 1] + log_a[:, j]
                best_prev = np.argmax(vals)
                backpointer[step, j] = best_prev
                viterbi_log[step, j] = vals[best_prev] + log_b[step, j]

        path = np.zeros(t, dtype=int)
        path[-1] = np.argmax(viterbi_log[-1])
        for step in range(t - 2, -1, -1):
            path[step] = backpointer[step + 1, path[step + 1]]

        return path
