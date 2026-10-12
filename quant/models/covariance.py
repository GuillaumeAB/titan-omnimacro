"""
quant.models.covariance — Robust covariance estimation.
Implements Ledoit-Wolf shrinkage, EWMA covariance, and nearest PSD projection.
"""

import numpy as np


def nearest_psd(C: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Projects a symmetric matrix to the nearest positive semi-definite matrix
    by clipping negative eigenvalues and restoring unit diagonals if applicable.
    """
    C = (C + C.T) / 2.0
    vals, vecs = np.linalg.eigh(C)
    vals = np.maximum(vals, eps)
    C_psd = vecs @ np.diag(vals) @ vecs.T
    return (C_psd + C_psd.T) / 2.0


def ewma_cov(X: np.ndarray, halflife: float = 26.0) -> np.ndarray:
    """
    Exponentially Weighted Moving Average (EWMA) covariance.
    X: shape (T, N)
    halflife: decay parameter in number of periods.
    """
    T, N = X.shape
    decay = 2.0 ** (-1.0 / halflife)
    weights = decay ** np.arange(T - 1, -1, -1)
    weights /= weights.sum()

    # Weighted mean
    mean = np.sum(X * weights[:, None], axis=0)
    X_centered = X - mean

    # Weighted covariance
    weighted_X = X_centered * np.sqrt(weights[:, None])
    cov = weighted_X.T @ weighted_X
    return nearest_psd(cov)


def ledoit_wolf(X: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Ledoit-Wolf (2004) analytical shrinkage covariance estimator.
    Shrinks the sample covariance matrix towards a structured target:
    F = s^2 * I (scaled identity) or constant correlation.
    Here implements standard Ledoit-Wolf constant correlation shrinkage:
    Target F_ij = r_bar * sqrt(S_ii * S_jj) for i != j, and S_ii for i == j.
    Returns (shrunk_cov, shrinkage_intensity).
    """
    T, N = X.shape
    if T <= 1 or N <= 1:
        S = np.cov(X, rowvar=False) if T > 1 else np.eye(N)
        return nearest_psd(S), 0.0

    # Demean
    mean = np.mean(X, axis=0)
    X_c = X - mean

    # Sample covariance S (unbiased, T-1)
    S = (X_c.T @ X_c) / (T - 1)
    var = np.diag(S)
    std = np.sqrt(np.maximum(var, 1e-12))

    # Sample correlation matrix
    corr = S / (std[:, None] @ std[None, :])
    np.fill_diagonal(corr, 1.0)

    # Average correlation r_bar
    sum_corr = np.sum(corr) - N
    r_bar = sum_corr / (N * (N - 1)) if N > 1 else 0.0

    # Target matrix F
    F = r_bar * (std[:, None] @ std[None, :])
    np.fill_diagonal(F, var)

    # Compute shrinkage intensity delta*
    # pi-hat = sum of asymptotic variances of sample covariance elements
    Y = np.zeros((T, N, N))
    for t in range(T):
        Y[t] = np.outer(X_c[t], X_c[t]) - S

    pi_mat = np.sum(Y ** 2, axis=0) / T
    pi_hat = np.sum(pi_mat)

    # gamma-hat = ||S - F||_F^2
    gamma_hat = np.sum((S - F) ** 2)

    # rho-hat
    rho_diag = np.sum(np.diag(pi_mat))
    rho_off = 0.0
    for i in range(N):
        for j in range(N):
            if i != j:
                # Asymptotic covariance term
                y_ii = Y[:, i, i]
                y_jj = Y[:, j, j]
                y_ij = Y[:, i, j]
                term1 = np.sum(y_ii * y_ij) / T
                term2 = np.sum(y_jj * y_ij) / T
                term_cov = (std[j] / std[i]) * term1 + (std[i] / std[j]) * term2
                rho_off += (r_bar / 2.0) * term_cov

    rho_hat = rho_diag + rho_off

    # Optimal shrinkage intensity kappa
    kappa = (pi_hat - rho_hat) / gamma_hat if gamma_hat > 1e-12 else 0.0
    shrinkage = max(0.0, min(1.0, kappa / T))

    # Shrunk covariance
    shrunk_cov = shrinkage * F + (1.0 - shrinkage) * S
    return nearest_psd(shrunk_cov), float(shrinkage)
