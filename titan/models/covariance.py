"""
titan.models.covariance — Robust covariance estimation.
Implements Ledoit-Wolf constant-correlation shrinkage, EWMA covariance,
and nearest positive semi-definite (PSD) projection.
"""

from __future__ import annotations

import numpy as np


def nearest_psd(c: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Projects a symmetric matrix to the nearest positive semi-definite matrix
    by clipping negative eigenvalues and restoring symmetry.
    """
    c_sym = (c + c.T) / 2.0
    vals, vecs = np.linalg.eigh(c_sym)
    vals = np.maximum(vals, eps)
    c_psd = vecs @ np.diag(vals) @ vecs.T
    return (c_psd + c_psd.T) / 2.0


def ewma_cov(x: np.ndarray, halflife: float = 26.0) -> np.ndarray:
    """
    Exponentially Weighted Moving Average (EWMA) covariance.
    x: shape (T, N)
    halflife: decay parameter in number of periods.
    """
    t, _ = x.shape
    decay = 2.0 ** (-1.0 / halflife)
    weights = decay ** np.arange(t - 1, -1, -1)
    weights /= weights.sum()

    mean = np.sum(x * weights[:, None], axis=0)
    x_centered = x - mean
    weighted_x = x_centered * np.sqrt(weights[:, None])
    cov = weighted_x.T @ weighted_x
    return nearest_psd(cov)


def ledoit_wolf(x: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Ledoit-Wolf (2004) analytical constant-correlation shrinkage estimator.
    Target F_ij = r_bar * sqrt(S_ii * S_jj) for i != j, and S_ii for i == j.
    Returns (shrunk_cov, shrinkage_intensity).
    """
    t, n = x.shape
    if t <= 1 or n <= 1:
        s = np.cov(x, rowvar=False) if t > 1 else np.eye(n)
        return nearest_psd(s), 0.0

    mean = np.mean(x, axis=0)
    x_c = x - mean

    s = (x_c.T @ x_c) / (t - 1)
    var = np.diag(s)
    std = np.sqrt(np.maximum(var, 1e-12))

    corr = s / (std[:, None] @ std[None, :])
    np.fill_diagonal(corr, 1.0)

    sum_corr = np.sum(corr) - n
    r_bar = sum_corr / (n * (n - 1)) if n > 1 else 0.0

    f = r_bar * (std[:, None] @ std[None, :])
    np.fill_diagonal(f, var)

    y = np.zeros((t, n, n))
    for step in range(t):
        y[step] = np.outer(x_c[step], x_c[step]) - s

    pi_mat = np.sum(y**2, axis=0) / t
    pi_hat = np.sum(pi_mat)

    gamma_hat = np.sum((s - f) ** 2)

    rho_diag = np.sum(np.diag(pi_mat))
    rho_off = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                term1 = np.sum(y[:, i, i] * y[:, i, j]) / t
                term2 = np.sum(y[:, j, j] * y[:, i, j]) / t
                term_cov = (std[j] / std[i]) * term1 + (std[i] / std[j]) * term2
                rho_off += (r_bar / 2.0) * term_cov

    rho_hat = rho_diag + rho_off

    kappa = (pi_hat - rho_hat) / gamma_hat if gamma_hat > 1e-12 else 0.0
    shrinkage = max(0.0, min(1.0, kappa / t))

    shrunk_cov = shrinkage * f + (1.0 - shrinkage) * s
    return nearest_psd(shrunk_cov), float(shrinkage)
