"""
quant.models.regime — Gaussian Hidden Markov Model (HMM) for market regimes.
Pure numpy/scipy Baum-Welch implementation in log-space with multiple restarts,
vectorized emission densities, forward filter, smoother, Viterbi decoding,
and automatic state ordering (State 0 = Calmest, State K-1 = Most Stressed).
"""

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
        max_iter: int = 200,
        tol: float = 1e-6,
        seed: int = 42,
        cov_floor: float = 1e-6
    ):
        self.n_states = n_states
        self.n_restarts = n_restarts
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.cov_floor = cov_floor

        self.startprob_: np.ndarray = None       # (K,)
        self.transmat_: np.ndarray = None        # (K, K)
        self.means_: np.ndarray = None           # (K, D)
        self.covars_: np.ndarray = None          # (K, D, D)
        self.loglik_: float = -np.inf
        self.state_order_: np.ndarray = None

    @property
    def state_names(self) -> list[str]:
        if self.n_states == 2:
            return ["Calm", "Stress"]
        elif self.n_states == 3:
            return ["Calm", "Normal", "Stress"]
        return [f"Regime {i}" for i in range(self.n_states)]

    def _init_params(self, X: np.ndarray, rng: np.random.Generator):
        T, D = X.shape
        K = self.n_states

        # k-means++ style mean initialization
        means = np.zeros((K, D))
        idx = rng.choice(T)
        means[0] = X[idx]

        for k in range(1, K):
            dists = np.min([np.sum((X - means[j]) ** 2, axis=1) for j in range(k)], axis=0)
            probs = dists / np.sum(dists)
            idx = rng.choice(T, p=probs)
            means[k] = X[idx]

        # Initial covariances = sample covariance
        base_cov = np.cov(X, rowvar=False)
        if D == 1:
            base_cov = np.array([[float(base_cov)]])
        covars = np.zeros((K, D, D))
        for k in range(K):
            covars[k] = base_cov + np.eye(D) * self.cov_floor

        # Transition matrix (high self-transition persistence)
        trans = np.full((K, K), 0.05 / (K - 1) if K > 1 else 1.0)
        np.fill_diagonal(trans, 0.95)
        startprob = np.ones(K) / K

        return startprob, trans, means, covars

    def _compute_log_emission(self, X: np.ndarray, means: np.ndarray, covars: np.ndarray) -> np.ndarray:
        T, D = X.shape
        K = self.n_states
        log_B = np.zeros((T, K))
        const = -0.5 * D * np.log(2.0 * np.pi)

        for k in range(K):
            diff = X - means[k]
            cov_k = covars[k] + np.eye(D) * self.cov_floor
            try:
                sign, logdet = np.linalg.slogdet(cov_k)
                if sign <= 0:
                    cov_k += np.eye(D) * 1e-4
                    sign, logdet = np.linalg.slogdet(cov_k)
                inv_cov = np.linalg.pinv(cov_k)
                quad = np.sum((diff @ inv_cov) * diff, axis=1)
                log_B[:, k] = const - 0.5 * logdet - 0.5 * quad
            except Exception:
                log_B[:, k] = -1e10

        return log_B

    def _forward_backward(self, log_B: np.ndarray, startprob: np.ndarray, transmat: np.ndarray):
        T, K = log_B.shape
        log_pi = np.log(np.maximum(startprob, 1e-15))
        log_A = np.log(np.maximum(transmat, 1e-15))

        # Forward pass (log-alpha)
        log_alpha = np.zeros((T, K))
        log_alpha[0] = log_pi + log_B[0]
        for t in range(1, T):
            for j in range(K):
                log_alpha[t, j] = logsumexp(log_alpha[t - 1] + log_A[:, j]) + log_B[t, j]

        loglik = logsumexp(log_alpha[-1])

        # Backward pass (log-beta)
        log_beta = np.zeros((T, K))
        log_beta[-1] = 0.0
        for t in range(T - 2, -1, -1):
            for i in range(K):
                log_beta[t, i] = logsumexp(log_A[i, :] + log_B[t + 1, :] + log_beta[t + 1, :])

        # Posterior state probabilities: gamma_t(i) = P(s_t = i | X)
        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)

        # Transition posteriors: xi_t(i, j) = P(s_t = i, s_{t+1} = j | X)
        log_xi = np.zeros((T - 1, K, K))
        for t in range(T - 1):
            term = log_alpha[t, :, None] + log_A + log_B[t + 1, None, :] + log_beta[t + 1, None, :]
            log_xi[t] = term - logsumexp(term)
        xi = np.exp(log_xi)

        return loglik, log_alpha, gamma, xi

    def fit(self, X: np.ndarray):
        """Fit Gaussian HMM with multiple random restarts."""
        X = np.asarray(X, dtype=float)
        T, D = X.shape
        K = self.n_states

        best_ll = -np.inf
        best_params = None

        master_rng = np.random.default_rng(self.seed)

        for restart in range(self.n_restarts):
            rng = np.random.default_rng(master_rng.integers(100000))
            pi, A, mu, sigma = self._init_params(X, rng)
            prev_ll = -np.inf

            for it in range(self.max_iter):
                log_B = self._compute_log_emission(X, mu, sigma)
                ll, _, gamma, xi = self._forward_backward(log_B, pi, A)

                if np.isnan(ll):
                    break

                if abs(ll - prev_ll) < self.tol:
                    break
                prev_ll = ll

                # M-step updates
                pi = gamma[0] / np.sum(gamma[0])
                A = np.sum(xi, axis=0) / np.maximum(np.sum(gamma[:-1], axis=0)[:, None], 1e-12)
                A = A / np.sum(A, axis=1, keepdims=True)

                gamma_sum = np.maximum(np.sum(gamma, axis=0), 1e-12)  # (K,)
                mu = (gamma.T @ X) / gamma_sum[:, None]

                for k in range(K):
                    diff = X - mu[k]
                    sigma[k] = (diff.T @ (diff * gamma[:, k, None])) / gamma_sum[k]
                    sigma[k] += np.eye(D) * self.cov_floor

            if prev_ll > best_ll:
                best_ll = prev_ll
                best_params = (pi, A, mu, sigma)

        if best_params is None:
            raise RuntimeError("Gaussian HMM failed to converge across all restarts.")

        pi, A, mu, sigma = best_params

        # Re-order states by ascending variance of column 0
        # (State 0 = Calmest, State K-1 = Highest volatility)
        vols = np.array([sigma[k, 0, 0] for k in range(K)])
        order = np.argsort(vols)
        self.state_order_ = order

        self.startprob_ = pi[order]
        self.transmat_ = A[order][:, order]
        self.means_ = mu[order]
        self.covars_ = sigma[order]
        self.loglik_ = best_ll
        return self

    def filter(self, X: np.ndarray) -> np.ndarray:
        """Forward filtered probabilities P(s_t | x_1..t). Shape (T, K)."""
        X = np.asarray(X, dtype=float)
        log_B = self._compute_log_emission(X, self.means_, self.covars_)
        T, K = log_B.shape
        log_pi = np.log(np.maximum(self.startprob_, 1e-15))
        log_A = np.log(np.maximum(self.transmat_, 1e-15))

        log_alpha = np.zeros((T, K))
        log_alpha[0] = log_pi + log_B[0]
        for t in range(1, T):
            for j in range(K):
                log_alpha[t, j] = logsumexp(log_alpha[t - 1] + log_A[:, j]) + log_B[t, j]

        # Normalise each step
        norm = log_alpha - logsumexp(log_alpha, axis=1, keepdims=True)
        return np.exp(norm)

    def smooth(self, X: np.ndarray) -> np.ndarray:
        """Smoothed state probabilities P(s_t | x_1..T). Shape (T, K)."""
        X = np.asarray(X, dtype=float)
        log_B = self._compute_log_emission(X, self.means_, self.covars_)
        _, _, gamma, _ = self._forward_backward(log_B, self.startprob_, self.transmat_)
        return gamma

    def viterbi(self, X: np.ndarray) -> np.ndarray:
        """Most likely state sequence (Viterbi path). Shape (T,)."""
        X = np.asarray(X, dtype=float)
        log_B = self._compute_log_emission(X, self.means_, self.covars_)
        T, K = log_B.shape
        log_pi = np.log(np.maximum(self.startprob_, 1e-15))
        log_A = np.log(np.maximum(self.transmat_, 1e-15))

        viterbi_log = np.zeros((T, K))
        backpointer = np.zeros((T, K), dtype=int)

        viterbi_log[0] = log_pi + log_B[0]
        for t in range(1, T):
            for j in range(K):
                vals = viterbi_log[t - 1] + log_A[:, j]
                best_prev = np.argmax(vals)
                backpointer[t, j] = best_prev
                viterbi_log[t, j] = vals[best_prev] + log_B[t, j]

        # Backtrack
        path = np.zeros(T, dtype=int)
        path[-1] = np.argmax(viterbi_log[-1])
        for t in range(T - 2, -1, -1):
            path[t] = backpointer[t + 1, path[t + 1]]

        return path
