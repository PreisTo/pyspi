import math

import numpy as np
from numba import njit


@njit(fastmath=True)
def b_maxL(source_counts, t, C):
    """
    Maximum-likelihood background rate b >= 0.

    Parameters
    ----------
    source_counts : array, shape (n_pointings,)
        Expected SOURCE COUNTS.
        Exposure time has already been applied.

    t : array, shape (n_pointings,)
        Exposure times.

    C : array, shape (n_pointings,)
        Observed counts.

    Returns
    -------
    b : float
        Maximum-likelihood background rate.
    """

    eps = 1e-12
    n_p = len(source_counts)

    # Derivative of log-likelihood at b = 0:
    #
    # d logL / db
    # = sum_p [C_p * t_p / (source_counts_p + t_p*b) - t_p]
    #
    f0 = 0.0

    for p in range(n_p):
        f0 += C[p] * t[p] / (source_counts[p] + eps) - t[p]

    # Constrained maximum is b = 0
    if f0 <= 0.0:
        return 0.0

    # Find upper bracket
    lo = 0.0
    hi = 1.0

    for _ in range(200):

        f = 0.0

        for p in range(n_p):
            expected = source_counts[p] + t[p] * hi

            f += C[p] * t[p] / (expected + eps) - t[p]

        if f < 0.0:
            break

        hi *= 2.0

    # Bisection
    for _ in range(100):

        mid = 0.5 * (lo + hi)

        f = 0.0

        for p in range(n_p):
            expected = source_counts[p] + t[p] * mid

            f += C[p] * t[p] / (expected + eps) - t[p]

        if f > 0.0:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


@njit(fastmath=True)
def logLcore(
    source_counts,
    t_elapsed,
    counts,
):
    """
    Full profile Poisson log-likelihood.

    Parameters
    ----------
    source_counts : array, shape
        (n_pointings, n_detectors, n_energy_bins)

        Expected source COUNTS.
        Exposure time has already been applied.

    t_elapsed : array, shape
        (n_pointings, n_detectors)

        Exposure times. These are used only to scale the
        shared background rate.

    counts : array, shape
        (n_pointings, n_detectors, n_energy_bins)

        Observed counts.

    Returns
    -------
    logL : float
        Full Poisson log-likelihood, with the background
        rate profiled independently for each detector /
        energy-bin combination.
    """

    n_p = source_counts.shape[0]
    n_dets = source_counts.shape[1]
    n_bins = source_counts.shape[2]

    if n_p <= 1:
        raise ValueError("logLcore requires more than one pointing.")

    logL = 0.0

    # Temporary arrays across pointings
    m = np.empty(n_p)
    t = np.empty(n_p)
    C = np.empty(n_p)

    for d in range(n_dets):

        # Exposure for this detector
        for p in range(n_p):
            t[p] = t_elapsed[p, d]

        for e in range(n_bins):

            # Collect source expectation and data
            # across all pointings
            for p in range(n_p):
                m[p] = source_counts[p, d, e]
                C[p] = counts[p, d, e]

            # ML background rate
            b = b_maxL(m, t, C)

            # Full Poisson likelihood
            for p in range(n_p):

                expected_counts = m[p] + t[p] * b

                logL += (
                    C[p] * math.log(expected_counts + 1e-300)
                    - expected_counts
                    - math.lgamma(C[p] + 1.0)
                )

    return logL
