from functools import reduce
import logging
import os
import numpy as np
import math
from threeML.plugins.DispersionSpectrumLike import DispersionSpectrumLike

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
        for p in self._binned_spec_with_disp_dict:
            for v in self._binned_spec_with_disp_dict[p].values():
                self._binned_spec_with_disp_list.append(v)

        rsps = []
        for b in self._binned_spec_with_disp_list:
            rsps.append(b.response)
        self._responses = rsps
        self._len_pointings = len(self._binned_spec_with_disp_dict.keys())
        self._len_echans = len(self._responses[0].ebounds) - 1
        self._len_dets = len(next(iter(self._binned_spec_with_disp_dict.values())))

        self._t_elapsed = np.zeros((self._len_pointings, self._len_dets))
        for p in range(self._len_pointings):
            for d in range(self._len_dets):
                self._t_elapsed[p, d] = self._binned_spec_with_disp_list[
                    (p + 1) + d
                ].observed_spectrum.exposure
        self._counts = np.zeros((self._len_pointings, self._len_dets, self._len_echans))
        for p in range(self._len_pointings):
            for d in range(self._len_dets):
                self._counts[p, d] = self._binned_spec_with_disp_list[
                    (p + 1) + d
                ].observed_spectrum.counts

        self._monte_carlo_energies = self._responses[
            0
        ].monte_carlo_energies  # they have to be the same for all responses

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
        for p in range(self._len_pointings):
            for d in range(self._len_dets):
                for e in range(self._len_echans):
                    self._lgammas -= math.lgamma(self._counts[p, d, e] + 1.0)

    def set_model(self, likelihoodmodel):
        self._like_model = likelihoodmodel
        differential_flux, integral = self._get_diff_flux_and_integral(
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
            integral = self._get_diff_flux_and_integral(
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

    def display_model(self):
        pass

    @property
    def responses(self):
        return self._responses

    def get_model(self, precalc_fluxes=None):
        if precalc_fluxes is None:

            precalc_fluxes = self._integral_flux()

        models = np.empty((self._len_pointings, self._len_dets, self._len_echans))
        for p in range(self._len_pointings):
            for d in range(self._len_dets):
                models[p, d] = self._binned_spec_with_disp_list[(p + 1) * d].get_model(
                    precalc_fluxes
                )
        return models.reshape(self._len_pointings, self._len_dets, self._len_echans)

    def get_log_like(self):
        source_rates = self.get_model()
        return logLcore(source_rates, self._t_elapsed, self._counts, self._lgammas)

    def get_number_of_data_points(self):
        return self._len_pointings * self._len_dets * self._len_echans

    @classmethod
    def from_pointing_list(
        cls, name, pointing_list, energy_bins, mc_energies, dets=None
    ):

        # TODO: make sure the energy bins are the same for all the plugins

        dets = [get_live_dets_pointing(p, event_types="single") for p in pointing_list]

        common_dets = reduce(np.intersect1d, dets)
        bs_dict = {}
        for pointing in pointing_list:
            bs_dict[pointing] = {}
            path = os.path.join(
                os.environ.get("PYSPI", None),
                "pointing_data",
                pointing,
            )

            time_start, time_elapsed, energy_bins, counts_f = extract_pointing_info(
                path, pointing, energy_bins=energy_bins
            )
            for i, d in enumerate(common_dets):
                bs_dict[pointing][d] = SPIPointingPlugin.from_data(
                    time_start,
                    time_elapsed[i],
                    counts_f[i],
                    energy_bins,
                    mc_energies,
                    pointing,
                    d,
                )
        return cls(name=name, binned_spec_with_disp_dict=bs_dict)
