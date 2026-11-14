"""Regularize covariance matrices and evaluate multivariate Gaussian log densities."""

from __future__ import annotations

import math

import numpy as np


def regularize_covariance(
    covariance: np.ndarray,
    *,
    diagonal_regularization: float,
    shrinkage: float,
    min_variance: float,
) -> np.ndarray:
    """Return a float64 positive-definite covariance from a finite square matrix.

    Symmetrize, shrink toward a variance-floored diagonal, add positive diagonal
    regularization, and floor eigenvalues at min_variance. shrinkage is in [0, 1].
    """
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be a square matrix.")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("covariance must be finite.")
    if float(diagonal_regularization) <= 0 or float(min_variance) <= 0:
        raise ValueError("covariance regularizers must be positive.")
    if not 0.0 <= float(shrinkage) <= 1.0:
        raise ValueError("shrinkage must be in [0, 1].")
    symmetric = 0.5 * (covariance + covariance.T)
    diagonal = np.maximum(np.diag(symmetric), float(min_variance))
    target = np.diag(diagonal)
    shrunk = (1.0 - float(shrinkage)) * symmetric + float(shrinkage) * target
    shrunk = shrunk + float(diagonal_regularization) * np.eye(shrunk.shape[0])
    (eigenvalues, eigenvectors) = np.linalg.eigh(shrunk)
    eigenvalues = np.maximum(eigenvalues, float(min_variance))
    # Scale eigenvector columns to reconstruct the covariance from its spectrum.
    return (eigenvectors * eigenvalues) @ eigenvectors.T


def gaussian_log_prob(
    value: np.ndarray, mean: np.ndarray, covariance: np.ndarray
) -> float:
    """Return log density for same-shaped vectors and their covariance matrix.

    Nonfinite inputs, a nonpositive determinant, or a failed solve return NaN;
    mismatched shapes raise ValueError. Callers supply a regularized covariance.
    """
    value = np.asarray(value, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    if value.ndim != 1 or value.shape != mean.shape:
        raise ValueError("value and mean must be same-shaped vectors.")
    if covariance.shape != (value.size, value.size):
        raise ValueError("covariance shape does not match value dimension.")
    if not (
        np.all(np.isfinite(value))
        and np.all(np.isfinite(mean))
        and np.all(np.isfinite(covariance))
    ):
        return float("nan")
    (sign, log_determinant) = np.linalg.slogdet(covariance)
    if sign <= 0:
        return float("nan")
    difference = value - mean
    try:
        quadratic = float(difference @ np.linalg.solve(covariance, difference))
    except np.linalg.LinAlgError:
        return float("nan")
    dimension = value.size
    return float(
        -0.5
        * (dimension * math.log(2.0 * math.pi) + float(log_determinant) + quadratic)
    )
