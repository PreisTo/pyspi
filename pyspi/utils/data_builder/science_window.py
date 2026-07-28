import copy
import os
import numpy as np
import logging

from astropy.io import fits
from pyspi.io.get_files import get_files
from pyspi.io.package_data import get_path_of_external_data_dir
from pyspi.utils.function_utils import (
    ISDC_MJD_to_cxcsec,
    find_needed_ids,
    get_time_object,
)
from pyspi.utils.livedets import double_names, get_live_dets, triple_names

log = logging.getLogger(__name__)


class SPISWFile:
    def __init__(self, det, pointing_id, ebounds):
        """
        Class to read in all the data needed from a SCW file
        for a given config file

        :param det: For which detector?
        :param pointing_id: Pointing ID
        :param ebounds: Energy bounds for the analysis
        """
        self._det_name = f"Detector {det}"
        self._mission = "Integral/SPI"
        self._ebounds = ebounds
        self._n_channels = len(self._ebounds) - 1

        if not isinstance(det, list):
            self._det = [det]
        else:
            self._det = det
        for d in self._det:
            assert d in np.arange(85), (
                f"{d} is not a valid"
                " detector. Please only use detector ids between 0 and 84."
            )

        self._pointing_id = pointing_id
        self._fetch_files()
        self._read_in_pointing_data(self._pointing_id)
        self._get_deadtime_info()

    def _fetch_files(self):
        """Get the data, either from afs or from ISDC archive."""
        try:
            get_files(self._pointing_id, access="afs")
        except AssertionError:
            print("AFS data access did not work." " I will try the ISDC data archive.")
            get_files(self._pointing_id, access="isdc")

    def _get_pointing_data_path(self, filename):
        """Helper to build path to a pointing data file."""
        return os.path.join(
            get_path_of_external_data_dir(),
            "pointing_data",
            self._pointing_id,
            filename,
        )

    def _bin_energies(self):
        """
        Assign energy channel numbers to events and discard events
        outside the defined energy bounds.
        """
        self._energy_bins = np.ones_like(self._energies, dtype=int) * -1
        for i, (emin, emax) in enumerate(zip(self._ebounds[:-1], self._ebounds[1:])):
            mask = np.logical_and(self._energies > emin, self._energies < emax)
            self._energy_bins[mask] = np.ones_like(self._energy_bins[mask]) * i

        mask = self._energy_bins == -1
        self._energy_bins = self._energy_bins[~mask]
        self._times = self._times[~mask]
        self._energies = self._energies[~mask]

    def _check_counts(self):
        """Raise if detector has zero counts."""
        if np.sum(self._energies) == 0:
            raise AssertionError(
                f"The detector {self._det} has zero counts"
                " and is therefore not active."
                " Please exclude this detector!"
            )

    def _read_ob_time_and_isdc_time(self):
        """
        Read OB-Time and ISDC-MJD time from the spi_oper file.

        :returns: (ob_time_data, time) arrays
        """
        with fits.open(self._get_pointing_data_path("spi_oper.fits.gz")) as hdu_oper:
            ob_time = hdu_oper["SPI.-OSGL-ALL"].data["OB_TIME"]
            time = hdu_oper["SPI.-OSGL-ALL"].data["TIME"]

        ob_time_data = (
            ob_time[:, 0] * 655363**3
            + ob_time[:, 1] * 655362**2
            + ob_time[:, 2] * 65536
            + ob_time[:, 3]
        )
        return ob_time_data, time

    def _read_deadtime_hk(self):
        """
        Read deadtime and OB-Time from the science housekeeping file.

        :returns: (deadtimes shaped (19, n_bins), ob_time_hk)
        """
        deadtimes = np.array([])
        with fits.open(
            self._get_pointing_data_path("spi_science_hk.fits.gz")
        ) as hdu_oper:
            for i in range(19):
                deadtimes = np.append(
                    deadtimes, hdu_oper["SPI.-SCHK-HRW"].data[f"P__DF__CAFDT__L{i}"]
                )
            ob_time = hdu_oper["SPI.-SCHK-CNV"].data["OB_TIME"]

        deadtimes = deadtimes.reshape(19, -1) * 100 * 10**-9

        ob_time_hk = (
            ob_time[:, 0] * 655363**3
            + ob_time[:, 1] * 655362**2
            + ob_time[:, 2] * 65536
            + ob_time[:, 3]
        )
        return deadtimes, ob_time_hk

    def _read_in_pointing_data(self, pointing_id):
        """
        Gets all needed information from the data file for the given
        pointing_id
        """
        with fits.open(self._get_pointing_data_path("spi_oper.fits.gz")) as hdu_oper:

            time_sgl = ISDC_MJD_to_cxcsec(hdu_oper[1].data["time"])
            time_psd = ISDC_MJD_to_cxcsec(hdu_oper[2].data["time"])
            time_me2 = ISDC_MJD_to_cxcsec(hdu_oper[4].data["time"])
            time_me3 = ISDC_MJD_to_cxcsec(hdu_oper[5].data["time"])

            time_start = np.min(
                np.concatenate([time_sgl, time_psd, time_me2, time_me3])
            )
            time_stop = np.max(np.concatenate([time_sgl, time_psd, time_me2, time_me3]))

            time_sgl -= time_start
            time_psd -= time_start
            time_me2 -= time_start
            time_me3 -= time_start
            self._time_start = 0
            self._time_stop = time_stop - time_start
            self._norm_time = time_start

            if self._det in range(19) or self._det == "singles":
                dets_sgl = hdu_oper[1].data["DETE"]
                if self._det != "singles":
                    time_sgl = time_sgl[dets_sgl == self._det]
                    energy_sgl = hdu_oper[1].data["energy"][dets_sgl == self._det]
                else:
                    energy_sgl = hdu_oper[1].data["energy"]

                dets_psd = hdu_oper[2].data["DETE"]
                if self._det != "singles":
                    time_psd = time_psd[dets_psd == self._det]
                    energy_psd = hdu_oper[2].data["energy"][dets_psd == self._det]
                else:
                    energy_psd = hdu_oper[2].data["energy"]

            if self._det in range(19, 61):
                dets_me2 = np.sort(hdu_oper[4].data["DETE"], axis=1)
                i, k = double_names[self._det]
                mask = np.logical_and(dets_me2[:, 0] == i, dets_me2[:, 1] == k)
                time_me2 = time_me2[mask]
                energy_me2 = np.sum(hdu_oper[4].data["energy"][mask], axis=1)

            if self._det in range(61, 85):
                dets_me3 = np.sort(hdu_oper[5].data["DETE"], axis=1)
                i, j, k = triple_names[self._det]
                mask = np.logical_and(
                    np.logical_and(dets_me3[:, 0] == i, dets_me3[:, 1] == j),
                    dets_me3[:, 2] == k,
                )
                time_me3 = time_me3[mask]
                energy_me3 = np.sum(hdu_oper[5].data["energy"][mask], axis=1)

        if self._det in range(19):
            self._times = time_psd
            self._energies = energy_psd
            self._times = np.append(self._times, time_sgl)
            self._energies = np.append(self._energies, energy_sgl)
            sort_array = np.argsort(self._times)
            self._times = self._times[sort_array]
            self._energies = self._energies[sort_array]

        if self._det in range(19, 61):
            self._times = time_me2
            self._energies = energy_me2

        if self._det in range(61, 85):
            self._times = time_me3
            self._energies = energy_me3

        self._check_counts()
        self._bin_energies()

    def _get_deadtime_info(self):
        """Get the deadtime info from the hk file."""
        ob_time_data, time = self._read_ob_time_and_isdc_time()
        deadtimes, ob_time_hk = self._read_deadtime_hk()

        times_hk = np.interp(ob_time_hk, ob_time_data, time)
        times_hk = np.append(times_hk, times_hk[-1] + (1.0 / (24 * 3600)))

        self._deadtime_bin_edges = ISDC_MJD_to_cxcsec(times_hk) - self._norm_time
        self._deadtimes = deadtimes[self._det]

    @property
    def geometry_file_path(self):
        """
        :returns: Path to the spacecraft geometry file
        """
        return self._get_pointing_data_path("sc_orbit_param.fits.gz")

    @property
    def times(self):
        """
        :returns: times of detected events
        """
        return self._times

    @property
    def energies(self):
        """
        :returns: energies of detected events
        """
        return self._energies

    @property
    def energy_bins(self):
        """
        :returns: energy bin number of every event
        """
        return self._energy_bins

    @property
    def ebounds(self):
        """
        :returns: ebounds of analysis
        """
        return self._ebounds

    @property
    def det(self):
        """
        :returns: detector ID
        """
        return self._det

    @property
    def n_channels(self):
        """
        :returns: number energy channels
        """
        return self._n_channels

    @property
    def time_start(self):
        """
        :returns: start time of lightcurve
        """
        return self._time_start

    @property
    def time_stop(self):
        """
        :returns: stop time of lightcurve
        """
        return self._time_stop

    @property
    def det_name(self):
        """
        :returns: Name det
        """
        return self._det_name

    @property
    def mission(self):
        """
        :returns: Name Mission
        """
        return self._mission

    @property
    def deadtime_bin_starts(self):
        """
        :returns: Start time of time bins which have the deadtime
        information
        """
        return self._deadtime_bin_edges[:-1]

    @property
    def deadtime_bin_stops(self):
        """
        :returns: Stop time of time bins which have the deadtime
        information
        """
        return self._deadtime_bin_edges[1:]

    @property
    def deadtimes_per_interval(self):
        """
        :returns: Deadtime per time bin which have the deadtime
        information
        """
        return self._deadtimes

    @property
    def livetimes_per_interval(self):
        """
        :returns: Livetime per time bin which have the deadtime
        information
        """
        return 1 - self._deadtimes


class SPISWFileGRB(SPISWFile):

    def __init__(self, det, ebounds, time_of_grb, sgl_type=None):
        """
        Class to read in all the data needed from a SCW file for a given
        grbtime

        :param det: For which detector?
        :param ebounds: Ebounds for the Analysis.
        :param time_of_grb: Time of the GRB as "YYMMDD HHMMSS"
        :param sgl_type: Which type of single events?
        Only normal sgl, psd or both?

        :returns: Object
        """
        # Store GRB-specific attributes before calling any base init logic
        self._time_of_GRB = get_time_object(time_of_grb)
        self._sgl_type = sgl_type

        self._det_name = f"Detector {det}"
        self._mission = "Integral/SPI"
        self._ebounds = ebounds
        self._n_channels = len(self._ebounds) - 1

        # Check that det is a valid number and filter non-live detectors
        if not isinstance(det, list):
            self._det = np.array([det])
        else:
            self._det = np.array(det)

        mask = []
        live_dets = get_live_dets(time_of_grb)
        for d in self._det:
            assert d in np.arange(85), (
                f"{d} is not a valid"
                " detector. Please only use detector ids between 0 and 84."
            )
            if d not in live_dets:
                print(f"Detector {d} removed because it was not working.")
                mask.append(False)
            else:
                mask.append(True)
        self._det = self._det[mask]

        if np.any(self._det < 19):
            assert sgl_type is not None, "Only PSD Events?"
            assert sgl_type in [
                "psd",
                "sgl",
                "both",
            ], "sgl_type must be psd, sgl or both"
            self._sgl_type = sgl_type

        # Find the SW ID of the data file we need for this time
        self._pointing_id = find_needed_ids(self._time_of_GRB)
        self._fetch_files()

        # Reference time of GRB
        self._GRB_ref_time_cxcsec = ISDC_MJD_to_cxcsec(
            (self._time_of_GRB).tt.mjd - 51544
        )

        # Read in all we need
        self._read_in_pointing_data()
        self._get_deadtime_info()
        self._get_gti_info()
        self._apply_gti_to_events()
        self._apply_gti_to_live_time()

    def _read_in_pointing_data(self):
        """
        Gets all needed information from the data file for the given
        pointing_id
        """
        with fits.open(self._get_pointing_data_path("spi_oper.fits.gz")) as hdu_oper:

            time_sgl = (
                ISDC_MJD_to_cxcsec(hdu_oper[1].data["time"]) - self._GRB_ref_time_cxcsec
            )
            time_psd = (
                ISDC_MJD_to_cxcsec(hdu_oper[2].data["time"]) - self._GRB_ref_time_cxcsec
            )
            time_me2 = (
                ISDC_MJD_to_cxcsec(hdu_oper[4].data["time"]) - self._GRB_ref_time_cxcsec
            )
            time_me3 = (
                ISDC_MJD_to_cxcsec(hdu_oper[5].data["time"]) - self._GRB_ref_time_cxcsec
            )

            self._time_start = np.min(
                np.concatenate([time_sgl, time_psd, time_me2, time_me3])
            )
            self._time_stop = np.max(
                np.concatenate([time_sgl, time_psd, time_me2, time_me3])
            )

            self._times = np.array([])
            self._energies = np.array([])

            for d in self._det:
                if d in range(19) or d == -1:
                    if self._sgl_type == "psd" or self._sgl_type == "both":
                        dets_sgl = hdu_oper[1].data["DETE"]
                        self._times = np.append(self._times, time_sgl[dets_sgl == d])
                        self._energies = np.append(
                            self._energies, hdu_oper[1].data["energy"][dets_sgl == d]
                        )

                    if self._sgl_type == "sgl" or self._sgl_type == "both":
                        dets_psd = hdu_oper[2].data["DETE"]
                        self._times = np.append(self._times, time_psd[dets_psd == d])
                        self._energies = np.append(
                            self._energies, hdu_oper[2].data["energy"][dets_psd == d]
                        )

                if d in range(19, 61):
                    dets_me2 = np.sort(hdu_oper[4].data["DETE"], axis=1)
                    i, k = double_names[d]
                    mask = np.logical_and(dets_me2[:, 0] == i, dets_me2[:, 1] == k)
                    self._times = np.append(self._times, time_me2[mask])
                    self._energies = np.append(
                        self._energies, np.sum(hdu_oper[4].data["energy"][mask], axis=1)
                    )

                if d in range(61, 85):
                    dets_me3 = np.sort(hdu_oper[5].data["DETE"], axis=1)
                    i, j, k = triple_names[d]
                    mask = np.logical_and(
                        np.logical_and(dets_me3[:, 0] == i, dets_me3[:, 1] == j),
                        dets_me3[:, 2] == k,
                    )
                    self._times = np.append(self._times, time_me3[mask])
                    self._energies = np.append(
                        self._energies, np.sum(hdu_oper[5].data["energy"][mask], axis=1)
                    )

            sort_array = np.argsort(self._times)
            self._times = self._times[sort_array]
            self._energies = self._energies[sort_array]

        self._check_counts()
        self._bin_energies()

    def _get_gti_info(self):
        with fits.open(self._get_pointing_data_path("spi_gti.fits.gz")) as hdu_oper:
            self._gti_start = (
                ISDC_MJD_to_cxcsec(hdu_oper[2].data["START"])
                - self._GRB_ref_time_cxcsec
            )
            self._gti_stop = (
                ISDC_MJD_to_cxcsec(hdu_oper[2].data["STOP"]) - self._GRB_ref_time_cxcsec
            )

    def _get_deadtime_info(self):
        """Get the deadtime info from the hk file."""
        ob_time_data, time = self._read_ob_time_and_isdc_time()
        deadtimes, ob_time_hk = self._read_deadtime_hk()

        # Fix invalid deadtimes > 1
        for d in range(19):
            idx = np.argwhere(deadtimes[d] > 1).flatten()
            if len(idx) > 0:
                for i in idx:
                    subarray = deadtimes[d, :i:-1]
                    val = subarray[subarray < 1].flatten()[0]
                    deadtimes[d, i] = val

        assert np.all(deadtimes < 1), "Deadtimes larger than 1 are not possible."

        times_hk = np.interp(ob_time_hk, ob_time_data, time)
        times_hk = np.append(times_hk, times_hk[-1] + (1.0 / (24 * 3600)))

        self._deadtime_bin_edges = (
            ISDC_MJD_to_cxcsec(times_hk) - self._GRB_ref_time_cxcsec
        )

        self._deadtimes = np.mean(deadtimes[self._det], axis=0)

        # Cut off data outside deadtime bin edges
        mask = np.logical_or(
            self._times < self._deadtime_bin_edges[0],
            self._times > self._deadtime_bin_edges[-1],
        )
        self._times = self._times[~mask]
        self._energies = self._energies[~mask]
        self._energy_bins = self._energy_bins[~mask]

        self._livetimes = 1 - self._deadtimes

    def _add_time_bins(self, starts, stops, max_time=1):

        save_livetimes = copy.deepcopy(self._livetimes)
        for i, (start, stop) in enumerate(zip(starts, stops)):
            if stop - start < max_time or max_time == -1:
                idx_start = np.argwhere(start >= self._deadtime_bin_edges)[-1][0]
                idx_stop = np.argwhere(stop > self._deadtime_bin_edges)[-1][0]
                if idx_start == idx_stop:
                    timebin_width = (
                        self._deadtime_bin_edges[idx_start + 1]
                        - self._deadtime_bin_edges[idx_start]
                    )

                    self._deadtime_bin_edges = np.insert(
                        self._deadtime_bin_edges, idx_start + 1, stop
                    )
                    self._deadtime_bin_edges = np.insert(
                        self._deadtime_bin_edges, idx_start + 1, start
                    )

                    livetime_rate_tot = self._livetimes[idx_start] / timebin_width
                    self._livetimes[idx_start] = livetime_rate_tot * (
                        self._deadtime_bin_edges[idx_start + 1]
                        - self._deadtime_bin_edges[idx_start]
                    )

                    self._livetimes = np.insert(
                        self._livetimes,
                        idx_start + 1,
                        livetime_rate_tot
                        * (
                            self._deadtime_bin_edges[idx_start + 3]
                            - self._deadtime_bin_edges[idx_start + 2]
                        ),
                    )
                    self._livetimes = np.insert(
                        self._livetimes,
                        idx_start + 1,
                        livetime_rate_tot
                        * (
                            self._deadtime_bin_edges[idx_start + 2]
                            - self._deadtime_bin_edges[idx_start + 1]
                        ),
                    )
                elif idx_start + 1 == idx_stop:
                    timebin_width1 = (
                        self._deadtime_bin_edges[idx_start + 1]
                        - self._deadtime_bin_edges[idx_start]
                    )
                    timebin_width2 = (
                        self._deadtime_bin_edges[idx_start + 2]
                        - self._deadtime_bin_edges[idx_start + 1]
                    )
                    old_edge = self._deadtime_bin_edges[idx_start + 1]
                    self._deadtime_bin_edges[idx_start + 1] = start
                    self._deadtime_bin_edges = np.insert(
                        self._deadtime_bin_edges, idx_start + 2, stop
                    )

                    livetime_rate_tot1 = self._livetimes[idx_start] / timebin_width1
                    livetime_rate_tot2 = self._livetimes[idx_start + 1] / timebin_width2

                    self._livetimes[idx_start] = livetime_rate_tot1 * (
                        self._deadtime_bin_edges[idx_start + 1]
                        - self._deadtime_bin_edges[idx_start]
                    )
                    self._livetimes[idx_start + 1] = livetime_rate_tot2 * (
                        self._deadtime_bin_edges[idx_start + 3]
                        - self._deadtime_bin_edges[idx_start + 2]
                    )
                    self._livetimes = np.insert(
                        self._livetimes,
                        idx_start + 1,
                        (
                            livetime_rate_tot1
                            * (old_edge - self._deadtime_bin_edges[idx_start + 1])
                            + livetime_rate_tot2
                            * (self._deadtime_bin_edges[idx_start + 2] - old_edge)
                        ),
                    )
                else:
                    timebin_width1 = (
                        self._deadtime_bin_edges[idx_start + 1]
                        - self._deadtime_bin_edges[idx_start]
                    )
                    livetime_rate_tot1 = self._livetimes[idx_start] / timebin_width1
                    new_livetime1 = livetime_rate_tot1 * (
                        start - self._deadtime_bin_edges[idx_start]
                    )

                    if idx_stop + 1 == len(self._deadtime_bin_edges):
                        new_livetimelast = 0
                        new_livetime_center = self._livetimes[idx_start] - new_livetime1
                        for idx in range(idx_start + 1, idx_stop):
                            new_livetime_center += self._livetimes[idx]
                        idx_stop -= 1
                    else:
                        timebin_widthlast = (
                            self._deadtime_bin_edges[idx_stop + 1]
                            - self._deadtime_bin_edges[idx_stop]
                        )
                        livetime_rate_totlast = (
                            self._livetimes[idx_stop] / timebin_widthlast
                        )
                        new_livetimelast = livetime_rate_totlast * (
                            self._deadtime_bin_edges[idx_stop + 1] - stop
                        )

                        new_livetime_center = (
                            self._livetimes[idx_start]
                            - new_livetime1
                            + self._livetimes[idx_stop]
                            - new_livetimelast
                        )
                        for idx in range(idx_start + 1, idx_stop):
                            new_livetime_center += self._livetimes[idx]

                    self._livetimes = np.delete(
                        self._livetimes, range(idx_start, idx_stop + 1)
                    )
                    self._deadtime_bin_edges = np.delete(
                        self._deadtime_bin_edges, range(idx_start + 1, idx_stop + 1)
                    )

                    self._livetimes = np.insert(
                        self._livetimes,
                        idx_start,
                        [new_livetime1, new_livetime_center, new_livetimelast],
                    )
                    self._deadtime_bin_edges = np.insert(
                        self._deadtime_bin_edges, idx_start + 1, [start, stop]
                    )

                mask = (
                    self._deadtime_bin_edges[1:] - self._deadtime_bin_edges[:-1]
                ) > 0

                deadtime_bin_edges_inter = self._deadtime_bin_edges[:-1][mask]
                self._deadtime_bin_edges = np.append(
                    deadtime_bin_edges_inter, self._deadtime_bin_edges[-1]
                )
                self._livetimes = self._livetimes[mask]

        assert np.isclose(np.sum(self._livetimes), np.sum(save_livetimes))

    def _apply_gti_to_live_time(self):

        self._add_time_bins(self._gti_start, self._gti_stop)

        bti_start = self._gti_stop[:-1]
        bti_stop = self._gti_start[1:]
        self._add_time_bins(bti_start, bti_stop, max_time=-1)

        for i, (start, stop) in enumerate(
            zip(self._deadtime_bin_edges[:-1], self._deadtime_bin_edges[1:])
        ):
            idx = np.argwhere(start >= self._gti_start)[-1][0]
            idx_stop = np.argwhere(stop >= self._gti_start)[-1][0]

            if start > self._gti_stop[idx]:
                frac_good = 0
            elif self._gti_stop[idx] >= stop:
                frac_good = 1
            elif idx_stop == idx:
                frac_good = (self._gti_stop[idx] - start) / (stop - start)
            elif idx_stop == idx + 1:
                frac_good = (self._gti_stop[idx] - start) / (stop - start)
                frac_good += (stop - self._gti_start[idx_stop]) / (stop - start)
            else:
                frac_good = (self._gti_stop[idx] - start) / (stop - start)
                frac_good += (stop - self._gti_start[idx_stop]) / (stop - start)
                for idx_loop in range(idx + 1, idx_stop):
                    frac_good += (
                        self._gti_stop[idx_loop] - self._gti_start[idx_loop]
                    ) / (stop - start)

            self._livetimes[i] *= frac_good

        return self._livetimes

    def _apply_gti_to_events(self):
        """
        Create a filter index for events falling outside of the GTI.
        """
        filter_idx = np.zeros_like(self._times, dtype=bool)
        for start, stop in zip(self._gti_start, self._gti_stop):
            tmp_idx = np.logical_and(start <= self._times, self._times <= stop)
            filter_idx = np.logical_or(filter_idx, tmp_idx)
        self._filter_idx = filter_idx

    @property
    def gti_times(self):
        """
        :returns: times of detected events within a gti
        """
        return self._times[self._filter_idx]

    @property
    def gti_energies(self):
        """
        :returns: energies of detected events
        """
        return self._energies[self._filter_idx]

    @property
    def gti_energy_bins(self):
        """
        :returns: energy bin number of every event
        """
        return self._energy_bins[self._filter_idx]

    @property
    def livetimes_per_interval(self):
        """
        :returns: Livetime per time bin which have the deadtime
        information
        """
        return self._livetimes
