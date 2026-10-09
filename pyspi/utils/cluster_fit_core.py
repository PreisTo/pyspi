import math
import os
from datetime import datetime

import astropy.time as at
import numpy as np
from astropy.io import fits
from astropy.table import Table
from numba import njit

from pyspi.io.get_files import get_files
from pyspi.utils.data_builder.science_window import SPISWFile
from pyspi.utils.response.spi_drm import SPIDRM


@njit
def b_maxL_1_flat(m_row, t, C_row):
    n = m_row.shape[0]
    eps = 1e-12
    s0 = 0.0
    for i in range(n):
        s0 += C_row[i] / (m_row[i] + eps)
    f0 = s0 - t * n
    if f0 <= 0.0:
        return 0.0
    lo = 0.0
    hi = 1.0
    for _ in range(60):
        s = 0.0
        for i in range(n):
            s += C_row[i] / (m_row[i] + hi + eps)
        if (s - t * n) < 0.0:
            break
        hi *= 2.0

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        s = 0.0
        for i in range(n):
            s += C_row[i] / (m_row[i] + mid + eps)
        if (s - t * n) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@njit
def b_maxL_n(m, t, C, n_p):
    eps = 1e-12
    t_tot = 0.0
    for i in range(n_p):
        t_tot += t[i]
    s0 = 0.0
    for i in range(n_p):
        s0 += C[i] / (m[i] + eps)
    if (s0 - t_tot) <= 0.0:
        return 0.0
    lo = 0.0
    hi = 1.0
    for _ in range(200):
        s = 0.0
        for i in range(n_p):
            s += C[i] / (m[i] + hi + eps)
        if (s - t_tot) < 0.0:
            break
        hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        s = 0.0
        for i in range(n_p):
            s += C[i] / (m[i] + mid + eps)
        if (s - t_tot) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@njit
def logLcore(
    spec_binned,
    pointings,
    dets,
    resp_mats,
    num_sources,
    t_elapsed,
    counts,
):
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
            if n_p == 1:
                m_row = m[0, :]
                t = t_b[0]
                n_bins = m_row.shape[0]
                C_row = np.zeros(n_bins)
                for e_i in range(n_bins):
                    C_row[e_i] = counts[p_i][0][d_i, e_i]
                b = b_maxL_1_flat(m_row, t, C_row)
                for e_i in range(n_bins):
                    logL += C_row[e_i] * math.log(t * (m_row[e_i] + b) + 1e-300) - t * (
                        m_row[e_i] + b
                    )
                continue
            for e_i in range(len(m[0])):
                m_b = m[:, e_i]
                for C_i in range(n_p):
                    C_b[C_i] = counts[p_i][C_i][d_i, e_i]
                if n_p in (2, 3, 4):
                    b = b_maxL_n(m_b, t_b, C_b, n_p)
                else:
                    return 0.0
                for m_i in range(n_p):
                    rate = t_elapsed[p_i][m_i][d_i] * (m[m_i, e_i] + b)
                    logL += counts[p_i][m_i][d_i, e_i] * math.log(rate + 1e-300) - rate
    return logL


def extract_pointing_info(path, p_id, energy_bins=None):
    # TODO: make a non all dets version
    num_dets = 19
    if os.path.exists(f"{path}/pointing.fits"):
        with fits.open(f"{path}/pointing.fits") as file:
            t = Table.read(file[1])
            index = np.argwhere(t["PTID_ISOC"] == p_id[:8])
            if len(index) < 1:
                raise KeyError(f"{p_id} not found")
            pointing_info = t[index[-1][0]]
            t1 = at.Time(f"{pointing_info['TSTART'] + 2451544.5}", format="jd").datetime
            time_start = datetime.strftime(t1, "%y%m%d %H%M%S")

        with fits.open(f"{path}/dead_time.fits") as file:
            t = Table.read(file[1])
            time_elapsed = np.zeros(num_dets)
            for i in range(num_dets):
                for j in index:
                    time_elapsed[i] += t["LIVETIME"][j[0] * 85 + i]

        with fits.open(f"{path}/energy_boundaries.fits") as file:
            t = Table.read(file[1])
            energy_bins = np.append(t["E_MIN"], t["E_MAX"][-1])

        with fits.open(f"{path}/evts_det_spec.fits") as file:
            t = Table.read(file[1])
            counts = np.zeros((num_dets, len(energy_bins) - 1))
            for i in range(num_dets):
                for j in index:
                    counts[i, :] += t["COUNTS"][j[0] * 85 + i]
        return time_start, time_elapsed, energy_bins, counts

    oper_file = f"{path}/spi_oper.fits.gz"
    if not os.path.exists(oper_file):
        get_files(p_id, access="isla")

    assert os.path.exists(oper_file), f"{oper_file} does not exist"

    if os.path.exists(oper_file):
        gti_path = None
        for fname in ["sc_gti.fits.gz", "sc_gti.fits", "spi_gti.fits.gz"]:
            candidate = f"{path}/{fname}"
            if os.path.exists(candidate):
                gti_path = candidate
                break
        if gti_path is None:
            raise Exception(f"{p_id}: no GTI file found to get start time")

        with fits.open(gti_path) as hdul:
            tstart = tstop = None
            for ext in hdul:
                cols = getattr(ext, "columns", None)
                if cols is not None and "START" in cols.names and len(ext.data) > 0:
                    tstart = float(ext.data["START"][0])
                    tstop = float(ext.data["STOP"][-1])
                    break
            if tstart is None:
                raise Exception(f"{p_id}: no START/STOP rows in {gti_path}")

        t1 = at.Time(tstart + 2451544.5, format="jd").datetime
        time_start = datetime.strftime(t1, "%y%m%d %H%M%S")
        duration_seconds = (tstop - tstart) * 86400.0
        time_elapsed = np.full(num_dets, duration_seconds)
        counts = np.zeros((num_dets, len(energy_bins) - 1))

        with fits.open(oper_file) as hdul:
            for hdu_index in (1, 2):
                if hdu_index >= len(hdul):
                    continue
                data = hdul[hdu_index].data
                if data is None or "DETE" not in data.columns.names:
                    continue
                dete = np.array(data["DETE"])
                energy = np.array(data["ENERGY"])
                for d in range(num_dets):
                    mask = dete == d
                    if np.any(mask):
                        hist, _ = np.histogram(energy[mask], bins=energy_bins)
                        counts[d, :] += hist
        assert energy_bins is not None, (
            "When not using spimodfit data you need to provide the energy_bins"
        )
        return time_start, time_elapsed, energy_bins, counts

    elif path is None:
        raise NotImplementedError()
        SPISWFile()

    return (
        None,
        np.zeros(num_dets),
        energy_bins,
        np.zeros((num_dets, len(energy_bins) - 1)),
    )


def generate_resp_mat(rmfs, len_dets, len_ebs, len_emod, ra, dec):
    sds = np.empty(0)
    for d in range(len_dets):
        sd = SPIDRM(rmfs[d], ra, dec)
        sds = np.append(sds, sd.matrix.T)
    return sds.reshape((len_dets, len_emod - 1, len_ebs - 1))


@njit
def powerlaw_binned_spectrum(energy_bins, spectrum):
    assert np.amin(energy_bins) > 0, "All energy bin values must be greater 0"
    assert np.amin(spectrum) > 0, "All spectrum values must be greater 0"
    B = np.log(spectrum[1:] / spectrum[:-1]) / np.log(
        energy_bins[1:] / energy_bins[:-1]
    )
    A = spectrum[:-1] / (energy_bins[:-1] ** B)
    C = B + 1.0
    C[np.abs(C) < 1e-7] = 0
    regular = np.nonzero(C)[0]
    non_regular = []
    for i in range(len(energy_bins) - 1):
        if i not in regular:
            non_regular.append(i)
    non_regular = np.array(non_regular)
    binned_spectrum = np.zeros(len(energy_bins) - 1)
    binned_spectrum[regular] = (
        A[regular]
        / (C[regular])
        * (
            energy_bins[regular + 1] ** (C[regular])
            - energy_bins[regular] ** (C[regular])
        )
    )
    binned_spectrum[non_regular] = A[non_regular] * (
        np.log(energy_bins[non_regular + 1]) - np.log(energy_bins[non_regular])
    )

    return binned_spectrum
