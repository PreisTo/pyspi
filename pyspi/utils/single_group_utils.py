import math

import numpy as np
from numba import njit


@njit(fastmath=True)
def b_maxL(source_counts, t, C):
    eps = 1e-12
    n_p = len(source_counts)
    f0 = 0.0
    for p in range(n_p):
        f0 += C[p] * t[p] / (source_counts[p] + eps) - t[p]

    if f0 <= 0.0:
        return 0.0

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


@njit(fastmath=True, cache=True)
def logLcore(source_counts, t_elapsed, counts, lgamma):

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
            for p in range(n_p):
                m[p] = source_counts[p, d, e]
                C[p] = counts[p, d, e]

            b = b_maxL(m, t, C)

            # Full Poisson likelihood
            for p in range(n_p):

                expected_counts = m[p] + t[p] * b

                logL += C[p] * math.log(expected_counts + 1e-300) - expected_counts

    return logL + lgamma
