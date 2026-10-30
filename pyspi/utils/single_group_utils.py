import math

import numpy as np
from numba import njit


@njit(fastmath=True, cache=True, inline="always")
def b_maxL(source_counts, t, C, sum_t):

    eps = 1e-12
    n_p = source_counts.shape[0]

    f0 = -sum_t

    for p in range(n_p):
        f0 += C[p] * t[p] / (source_counts[p] + eps)

    if f0 <= 0.0:
        return 0.0

    lo = 0.0
    hi = 1.0

    for _ in range(64):

        f = -sum_t

        for p in range(n_p):
            expected = source_counts[p] + t[p] * hi
            f += C[p] * t[p] / (expected + eps)

        if f <= 0.0:
            break

        hi *= 2.0

    b = 0.5 * (lo + hi)

    for _ in range(30):

        f = -sum_t
        df = 0.0

        for p in range(n_p):

            tp = t[p]
            cp = C[p]

            expected = source_counts[p] + tp * b + eps
            inv_expected = 1.0 / expected

            f += cp * tp * inv_expected
            df -= cp * tp * tp * inv_expected * inv_expected

        if f > 0.0:
            lo = b
        else:
            hi = b

        if hi - lo <= 1e-10 * (1.0 + b):
            break

        if df != 0.0:
            b_new = b - f / df
        else:
            b_new = 0.5 * (lo + hi)

        if b_new <= lo or b_new >= hi:
            b = 0.5 * (lo + hi)
        else:
            b = b_new

    return b


@njit(fastmath=True, cache=True)
def logLcore(source_counts, t_elapsed, counts, lgamma):

    n_dets, n_bins, n_p = source_counts.shape

    if n_p <= 1:
        raise ValueError("logLcore requires more than one pointing.")

    logL = 0.0

    for d in range(n_dets):

        t = t_elapsed[d]

        # Only calculate this once per detector
        sum_t = 0.0
        for p in range(n_p):
            sum_t += t[p]

        for e in range(n_bins):

            m = source_counts[d, e]
            C = counts[d, e]

            b = b_maxL(m, t, C, sum_t)

            local_logL = 0.0

            for p in range(n_p):

                expected = m[p] + t[p] * b

                local_logL -= expected

                if C[p] != 0.0:
                    local_logL += C[p] * math.log(expected + 1e-300)

            logL += local_logL

    return logL + lgamma
