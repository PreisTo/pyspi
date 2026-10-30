from functools import reduce
import logging
import os
import numpy as np
import math
from threeML.plugins.DispersionSpectrumLike import DispersionSpectrumLike
from threeML.io.plotting.data_residual_plot import ResidualPlot

from pyspi.utils.cluster_fit_core import extract_pointing_info
from pyspi.utils.single_group_utils import logLcore
from pyspi.utils.livedets import get_live_dets_pointing
from pyspi.utils.spi_disp import SPIPointingPlugin

log = logging.getLogger(__name__)


class DispersionSpectrumSetLike(DispersionSpectrumLike):
    """
    Class that holds multiple detectors!
    """

    def __init__(
        self,
        name,
        binned_spec_with_disp_dict,
        **kwargs,
    ):
        self._name = name
        self._binned_spec_with_disp_dict = binned_spec_with_disp_dict
        self._binned_spec_with_disp_list = []
        for d in self._binned_spec_with_disp_dict:  # detectors
            for v in self._binned_spec_with_disp_dict[d].values():  # pointings
                self._binned_spec_with_disp_list.append(v)

        rsps = []
        for b in self._binned_spec_with_disp_list:
            rsps.append(b.response)
        self._responses = rsps
        self._len_dets = len(self._binned_spec_with_disp_dict.keys())
        self._len_echans = len(self._responses[0].ebounds) - 1
        self._len_pointings = len(next(iter(self._binned_spec_with_disp_dict.values())))

        self._t_elapsed = np.empty(
            (
                self._len_dets,
                self._len_pointings,
            )
        )
        for i, d in enumerate(self._binned_spec_with_disp_dict.keys()):
            for j, p in enumerate(self._binned_spec_with_disp_dict[d].keys()):
                self._t_elapsed[i, j] = self._binned_spec_with_disp_dict[d][
                    p
                ].observed_spectrum.exposure
        self._counts = np.empty((self._len_dets, self._len_echans, self._len_pointings))
        temp_counts = np.empty((self._len_dets, self._len_pointings, self._len_echans))
        for i, d in enumerate(self._binned_spec_with_disp_dict.keys()):
            for j, p in enumerate(self._binned_spec_with_disp_dict[d].keys()):
                temp_counts[i, j] = self._binned_spec_with_disp_dict[d][
                    p
                ].observed_spectrum.counts

        self._counts = temp_counts.swapaxes(1, 2)

        self._monte_carlo_energies = self._responses[
            0
        ].monte_carlo_energies  # they have to be the same for all responses
        self._ebounds = self._responses[0].ebounds
        # they have to be the same for all responses

        # TODO: !!!
        self._nuisance_parameters = {}
        self._model_integrate_method = "simpson"
        self._source_name = None

        self._has_contiguous_energies: bool = np.all(
            [
                b.observed_spectrum.is_contiguous()
                for b in self._binned_spec_with_disp_list
            ]
        )
        self._tag = None
        self._stokes = None

        self._predefined_energies: np.ndarray = self._monte_carlo_energies

        self._lgammas = 0.0

        flat = self._counts.ravel()

        for i in range(flat.size):
            self._lgammas -= math.lgamma(flat[i] + 1.0)

    def set_model(self, likelihoodmodel):
        self._like_model = likelihoodmodel
        _, integral = self._get_diff_flux_and_integral(
            self._like_model, integrate_method=self._model_integrate_method
        )

        self._integral_flux = integral

        for bs in self._binned_spec_with_disp_list:
            bs.set_model(likelihoodmodel)

    def set_model_integrate_method(self, method: str):
        """Change the integrate method for the model integration :param method:

        (str) which method should be used (simpson or trapz)
        """
        if method not in ["simpson", "trapz", "riemann"]:
            raise RuntimeError(
                "Only simpson, trapz and riemann are valid intergate methods."
            )

        self._model_integrate_method = method
        log.info(f"{self._name} changing model integration method to {method}")

        # if like_model already set, upadte the integral function
        if self._like_model is not None:
            _, integral = self._get_diff_flux_and_integral(
                self._like_model, integrate_method=method
            )

            self._integral_flux = integral
            for b in self._binned_spec_with_disp_list:
                b.response.set_function(self._integral_flux)

    def display_rsp(self):
        figs = []
        for r in self.responses:
            figs.append(r.plot_matrix())
        return figs

    @property
    def responses(self):
        return self._responses

    def get_model(self, precalc_fluxes=None):
        if precalc_fluxes is None:

            precalc_fluxes = self._integral_flux()

        models = np.empty((self._len_dets, self._len_pointings, self._len_echans))
        for i, d in enumerate(self._binned_spec_with_disp_dict.keys()):
            for j, p in enumerate(self._binned_spec_with_disp_dict[d].keys()):
                models[i, j] = self._binned_spec_with_disp_dict[d][p].get_model(
                    precalc_fluxes
                )
        models = models.swapaxes(1, 2)
        return models

    def get_log_like(self):
        source_rates = self.get_model()
        logl = logLcore(source_rates, self._t_elapsed, self._counts, self._lgammas)
        if not np.isfinite(logl):
            return -1e100
        return logl

    def get_number_of_data_points(self):
        return self._len_pointings * self._len_dets * self._len_echans

    def display_model(self, *args, **kwargs):
        residual_plot = ResidualPlot(**kwargs)

        x = self._ebounds[:-1] + np.diff(self._ebounds[1:], self._ebounds[:-1]) / 2
        y = np.zeros_like(x)
        for i in range(self._len_pointings):
            for d in range(self._len_dets):
                rebinned_quants = self._binned_spec_with_disp_list[
                    (i + 1) * d
                ]._construct_counts_arrays(min_rate=1e-99)
                y += 0

                pass

        # residual_plot.add_data(x, y,None )

        pass

    @classmethod
    def from_pointing_list(
        cls, name, pointing_list, energy_bins, mc_energies, dets=None, pointing_pos=None
    ):

        # TODO: make sure the energy bins are the same for all the plugins

        dets = [get_live_dets_pointing(p, event_types="single") for p in pointing_list]

        common_dets = reduce(np.intersect1d, dets)
        bs_dict = {}
        for pointing in pointing_list:
            path = os.path.join(
                os.environ.get("PYSPI", None),
                "pointing_data",
                pointing,
            )

            time_start, time_elapsed, energy_bins, counts_f = extract_pointing_info(
                path, pointing, energy_bins=energy_bins
            )
            for i, d in enumerate(common_dets):
                if str(d) not in bs_dict:
                    bs_dict[str(d)] = {}
                bs_dict[str(d)][pointing] = SPIPointingPlugin.from_data(
                    time_start,
                    time_elapsed[i],
                    counts_f[i],
                    energy_bins,
                    mc_energies,
                    pointing_number=pointing,
                    det=d,
                    pointing_pos=pointing_pos,
                )
        return cls(name=name, binned_spec_with_disp_dict=bs_dict)
