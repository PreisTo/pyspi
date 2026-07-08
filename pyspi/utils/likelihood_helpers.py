import numpy as np
from numba import njit


@njit
def b_maxL_2(m, t, C):
    first = C[0] + C[1] - (m[0] + m[1]) * (t[0] + t[1])
    root = (C[0] + C[1] + (m[0] - m[1]) * (t[0] + t[1])) ** 2 - 4 * C[0] * (
        m[0] - m[1]
    ) * (t[0] + t[1])
    res = (first + np.sqrt(root)) / (2 * (t[0] + t[1]))
    return res


@njit
def b_maxL_3(m, t, C):
    mt = m[0] + m[1] + m[2]
    tt = t[0] + t[1] + t[2]
    Ct = C[0] + C[1] + C[2]
    a = -tt
    b = -tt * mt + Ct
    c = (
        Ct * mt
        - C[0] * m[0]
        - C[1] * m[1]
        - C[2] * m[2]
        - tt * (m[0] * m[1] + m[1] * m[2] + m[2] * m[0])
    )
    d = (
        C[0] * m[1] * m[2]
        + C[1] * m[2] * m[0]
        + C[2] * m[0] * m[1]
        - tt * m[0] * m[1] * m[2]
    )
    D0 = b**2 - 3 * a * c
    D1 = 2 * b**3 - 9 * a * b * c + 27 * (a**2) * d

    if D0 == 0.0 and D1 == 0.0:
        return -b / (3 * a)

    C0 = ((D1 + np.sqrt(D1**2 - 4 * D0**3 + 0j)) / 2) ** (1 / 3)

    if C0 == 0:
        C0 = ((D1 - np.sqrt(D1**2 - 4 * D0**3 + 0j)) / 2) ** (1 / 3)

    x0 = -1 / (3 * a) * (b + C0 + D0 / C0)

    return x0.real


@njit
def b_maxL_4(m, t, C):
    t_t = np.sum(t)
    C_ = np.zeros(4)
    m_plus = np.zeros(4)
    m_cross = np.zeros(4)
    m_dot = np.zeros(4)

    for i in range(4):
        C_[i] = C[i] - t[i] * m[i]
        m_plus[i] = m[(i + 1) % 4] + m[(i + 2) % 4] + m[(i + 3) % 4]
        m_cross[i] = (
            m[(i + 1) % 4] * m[(i + 2) % 4]
            + m[(i + 2) % 4] * m[(i + 3) % 4]
            + m[(i + 3) % 4] * m[(i + 1) % 4]
        )
        m_dot[i] = m[(i + 1) % 4] * m[(i + 2) % 4] * m[(i + 3) % 4]

    A = -t_t.item()
    B = np.sum(C_ - t * m_plus).item()
    C = np.sum(C_ * m_plus - t * m_cross).item()
    D = np.sum(C_ * m_cross - t * m_dot).item()
    E = np.sum(C_ * m_dot).item()

    # print(A,B,C,D,E)

    alpha = -3 * B**2 / (8 * A**2) + C / A
    beta = B**3 / (8 * A**3) - B * C / (2 * A**2) + D / A
    gamma = (
        -3 * B**4 / (256 * A**4) + C * B**2 / (16 * A**3) - B * D / (4 * A**2) + E / A
    )

    if beta == 0.0:
        s1 = (alpha**2 - 4 * gamma) ** 0.5
        s2 = ((-alpha + s1) / 2) ** 0.5
        x = -B / (4 * A) + s2
        return x

    P = -(alpha**2) / 12 - gamma
    Q = -(alpha**3) / 108 + alpha * gamma / 3 - (beta**2) / 8
    R = -Q / 2 + (Q**2 / 4 + P**3 / 27) ** 0.5
    U = R ** (1 / 3)
    if U == 0.0:
        y = -5 / 6 * alpha - (Q ** (1 / 3))
    else:
        y = -5 / 6 * alpha + U - P / (3 * U)
    W = (alpha + 2 * y) ** 0.5

    s1 = 2 * beta / W
    s2 = (-(3 * alpha + 2 * y + s1)) ** 0.5
    s3 = (W + s2) / 2
    x = -B / (4 * A) + s3
    return x.real


@njit
def logLcore(spec_binned, pointings, dets, resp_mats, num_sources, t_elapsed, counts):
    logL = 0
    for p_i in range(len(pointings)):
        for d_i in range(len(dets[p_i])):
            n_p = len(pointings[p_i])
            m = np.zeros((n_p, len(resp_mats[p_i][0][0][0, 0, :])))

            t_b = np.zeros(n_p)
            for t_i in range(n_p):
                t_b[t_i] = t_elapsed[p_i][t_i][d_i]
            C_b = np.zeros(n_p)

            for s_i in range(num_sources):
                for m_i in range(n_p):
                    m[m_i, :] += np.dot(
                        spec_binned[s_i, :], resp_mats[p_i][s_i][m_i][d_i]
                    )
            for e_i in range(len(m[0])):
                m_b = m[:, e_i]
                for C_i in range(n_p):
                    C_b[C_i] = counts[p_i][C_i][d_i, e_i]

                if n_p == 2:
                    b = b_maxL_2(m_b, t_b, C_b)
                elif n_p == 3:
                    b = b_maxL_3(m_b, t_b, C_b)
                elif n_p == 4:
                    b = b_maxL_4(m_b, t_b, C_b)
                else:
                    print()
                    print("b_maxL is not defined")
                    print()
                    return 0.0
                for m_i in range(n_p):
                    logL += counts[p_i][m_i][d_i, e_i] * np.log(
                        t_elapsed[p_i][m_i][d_i] * (m[m_i, e_i] + b)
                    ) - t_elapsed[p_i][m_i][d_i] * (m[m_i, e_i] + b)
    return logL
