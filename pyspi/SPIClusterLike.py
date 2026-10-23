import logging

import numpy as np
from threeML import PluginPrototype

from pyspi.utils.cluster_fit_core import (
    extract_pointing_info,
    generate_resp_mat,
    logLcore,
    powerlaw_binned_spectrum,
)
from pyspi.utils.function_utils import find_response_version
from pyspi.utils.livedets import get_live_dets
from pyspi.utils.response.spi_response import ResponseRMFGenerator
from pyspi.utils.rsp_bases import get_base_rsp

log = logging.getLogger(__name__)


class SPIClusterLike(PluginPrototype):
    def __init__(
        self,
        name,
        pointings,
        energy_range,
        emod,  # mc energies
        binning_func,
        source_spectrum_powerlaw_binning=True,
    ):

        super().__init__(name, {})
        self._pointings = pointings
        self._energy_range = energy_range
        self._emod = emod
        self._binning_func = binning_func
        self._source_spectrum_powerlaw_binning = source_spectrum_powerlaw_binning
        self._n_calls = 0
        self._n_success = 0
        self._n_naninf = 0
        self._exc_counts = {}
        self._source_model = None
        self._source = None
        self._prepare_fit_data()

    def set_model(self, likelihood_model_instance):
        self._source_model = likelihood_model_instance
        self._find_updatable_sources()
        self._initialize_resp_mats()

    def get_spec_binnded(self):
        if self._source is None:
            num_sources = len(self._source_model.sources)
        else:
            num_sources = 1  # TODO: generalize that

        spec_binned = np.zeros((num_sources, len(self._emod) - 1))
        if self._source is None:
            for i, source in enumerate(list(self._source_model.sources.values())):
                spec = source(self._emod)
                if self._source_spectrum_powerlaw_binning:
                    spec_binned[i, :] = powerlaw_binned_spectrum(self._emod, spec)
                else:
                    spec_binned[i, :] = (
                        (self._emod[1:] - self._emod[:-1]) * (spec[:-1] + spec[1:]) / 2
                    )
        else:
            i = 0
            spec = self._source_model.sources[self._source](self._emod)
            if self._source_spectrum_powerlaw_binning:
                spec_binned[i, :] = powerlaw_binned_spectrum(self._emod, spec)
            else:
                spec_binned[i, :] = (
                    (self._emod[1:] - self._emod[:-1]) * (spec[:-1] + spec[1:]) / 2
                )
        return spec_binned

    def get_log_like(self):
        self._n_calls += 1
        if self._source is None:
            num_sources = len(self._source_model.sources)
        else:
            num_sources = 1  # TODO: generalize that

        spec_binned = np.zeros((num_sources, len(self._emod) - 1))
        try:
            if self._source is None:
                for i, source in enumerate(list(self._source_model.sources.values())):
                    spec = source(self._emod)
                    if self._source_spectrum_powerlaw_binning:
                        spec_binned[i, :] = powerlaw_binned_spectrum(self._emod, spec)
                    else:
                        spec_binned[i, :] = (
                            (self._emod[1:] - self._emod[:-1])
                            * (spec[:-1] + spec[1:])
                            / 2
                        )
            else:
                i = 0
                spec = self._source_model.sources[self._source](self._emod)
                if self._source_spectrum_powerlaw_binning:
                    spec_binned[i, :] = powerlaw_binned_spectrum(self._emod, spec)
                else:
                    spec_binned[i, :] = (
                        (self._emod[1:] - self._emod[:-1]) * (spec[:-1] + spec[1:]) / 2
                    )

            if 1 in self._updatable_sources:
                self._update_resp_mats()

            logL = logLcore(
                spec_binned,
                self._pointings,
                self._dets,
                self._resp_mats,
                num_sources,
                self._t_elapsed,
                self._counts,
            )
            if not np.isfinite(logL):
                self._n_naninf += 1
                return -1e10
            self._n_success += 1
            return logL

        except Exception as e:
            key = f"{type(e).__name__}: {e}"
            self._exc_counts[key] = self._exc_counts.get(key, 0) + 1
            return -1e10

    def get_number_of_data_points(self):
        return np.prod(self._counts.shape)

    def report_call_tally(self):
        log.info("\n" + "=" * 70)
        log.info(f"DIAGNOSTIC: {self._name} get_log_like call tally")
        log.info(f"  total calls:        {self._n_calls}")
        log.info(f"  succeeded (finite): {self._n_success}")
        log.info(f"  NaN/inf (no exc.):  {self._n_naninf}")
        log.info(f"  exceptions:         {sum(self._exc_counts.values())}")
        if self._exc_counts:
            log.info("  exception breakdown:")
            for msg, count in sorted(self._exc_counts.items(), key=lambda kv: -kv[1]):
                log.info(f"    {count:>5}x  {msg}")
        if self._n_calls and self._n_success == 0:
            log.warning(
                "!!! WARNING: ZERO successful calls -- this fit's "
                "result is meaningless (pure prior), not a real "
                "measurement. Do not trust it."
            )
        log.info("=" * 70 + "\n")

    def inner_fit(self):
        return self.get_log_like()

    def _prepare_fit_data(self):
        ebs, counts, dets, t_elapsed, t_start = [], [], [], [], []
        for combination in self._pointings:
            c_time_start, c_time_elapsed = [], []
            for p_i, pointing in enumerate(combination):
                time_start, time_elapsed, energy_bins, counts_f = extract_pointing_info(
                    pointing[1], pointing[0]
                )
                c_time_start.append(time_start)
                dets_temp = get_live_dets(time=time_start, event_types=["single"])
                c_time_elapsed.append(time_elapsed[dets_temp])

                if p_i == 0:
                    dets_0 = dets_temp
                    energy_bins_0 = energy_bins
                    c_counts_f = counts_f[dets_0]

                else:
                    c_counts_f = np.append(c_counts_f, counts_f[dets_0], axis=0)
                    assert np.array_equal(dets_0, dets_temp), (
                        f"Active detectors are not the same for "
                        f"{combination[0][0]} and {pointing[0]}"
                    )
                    assert np.array_equal(energy_bins_0, energy_bins), (
                        f"Energy bins are not the same for "
                        f"{combination[0][0]} and {pointing[0]}"
                    )
            eb, c = self._binning_func(energy_bins_0, c_counts_f, self._energy_range)
            nd = len(dets_0)
            counts.append(
                tuple(c[i * nd : (i + 1) * nd] for i in range(len(combination)))
            )
            ebs.append(eb)
            t_start.append(tuple(c_time_start))
            dets.append(dets_0)
            t_elapsed.append(tuple(c_time_elapsed))

        self._ebs = np.array(ebs)
        self._counts = np.array(counts)
        self._dets = np.array(dets)
        self._t_elapsed = np.array(t_elapsed)
        self._t_start = np.array(t_start)

    def _initialize_resp_mats(self):
        resp_mats, rmfs = [], []
        for count, combination in enumerate(self._pointings):
            source_resp_mats = []
            dets = self._dets[count]
            ebs = self._ebs[count]
            for k, source in self._source_model.sources.items():
                if self._source is not None and self._source != k:
                    continue

                combination_resp_mats, combination_rmfs = [], []
                for pointing in range(len(combination)):
                    time = self._t_start[count][pointing]
                    version = find_response_version(time)
                    rsp_base = get_base_rsp(version)
                    pointing_rmfs = tuple(
                        ResponseRMFGenerator.from_time(
                            time, d, ebs, self._emod, rsp_base
                        )
                        for d in dets
                    )
                    combination_resp_mats.append(
                        generate_resp_mat(
                            pointing_rmfs,
                            len(dets),
                            len(ebs),
                            len(self._emod),
                            source.position.get_ra(),
                            source.position.get_dec(),
                        )
                    )
                    combination_rmfs.append(pointing_rmfs)
                source_resp_mats.append(tuple(combination_resp_mats))
            resp_mats.append(tuple(source_resp_mats))
            rmfs.append(tuple(combination_rmfs))
        self._resp_mats = tuple(resp_mats)
        if 1 in self._updatable_sources:
            self._updatable_rmfs = tuple(rmfs)

    def _update_resp_mats(self):
        for count, combination in enumerate(self._pointings):
            for source_num, source in enumerate(
                list(self._source_model.sources.values())
            ):
                if self._updatable_sources[source_num] == 1:
                    for pointing in range(len(combination)):
                        self._resp_mats[count][source_num][pointing][:, :, :] = (
                            generate_resp_mat(
                                self._updatable_rmfs[count][pointing],
                                len(self._dets[count]),
                                len(self._ebs[count]),
                                len(self._emod),
                                source.position.get_ra(),
                                source.position.get_dec(),
                            )
                        )

    def _find_updatable_sources(self):
        keywords = ["position"]
        if self._source is None:
            self._updatable_sources = np.zeros(len(self._source_model.sources), np.int8)
            for s_i, source in enumerate(list(self._source_model.sources.values())):
                for parameter in source.free_parameters.values():
                    first_pos = parameter.path.find(".")
                    second_pos = parameter.path.find(".", first_pos + 1)
                    if parameter.path[first_pos + 1 : second_pos] in keywords:
                        self._updatable_sources[s_i] = 1
        else:
            self._updatable_sources = np.zeros(1, np.int8)
            source = self._source_model.sources[self._source]
            for parameter in list(source.free_parameters.values()):
                first_pos = parameter.path.find(".")
                second_pos = parameter.path.find(".", first_pos + 1)
                if parameter.path[first_pos + 1 : second_pos] in keywords:
                    self._updatable_sources[0] = 1

    def assign_to_source(self, source_name):
        if self._source_model is not None:
            assert (
                source_name in self._source_model.sources
            ), f"{source_name} is not a known source name"
        self._source = source_name


class SPIClusterPlugin(SPIClusterLike):
    pass
