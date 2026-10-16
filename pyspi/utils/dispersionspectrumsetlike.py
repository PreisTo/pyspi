from functools import reduce
import logging
import os
import numpy as np
from threeML.plugins.DispersionSpectrumLike import DispersionSpectrumLike

from pyspi.utils.cluster_fit_core import extract_pointing_info, logLcore
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
        self._binned_spec_with_disp_dict = binned_spec_with_disp_dict
        self._binned_spec_with_disp_list = []
        for p in self._binned_spec_with_disp_dict:
            for v in self._binned_spec_with_disp_dict[p].values():
                self._binned_spec_with_disp_list.append(v)

        rsps = []
        for b in self._binned_spec_with_disp_list:
            rsps.append(b.response)
        self._responses = rsps

    def set_model(self, likelihoodmodel):
        for bs in self._binned_spec_with_disp_list:
            bs.set_model(likelihoodmodel)

    def _evaluate_model(self, precalc_fluxes: np.ndarray | None = None):
        fluxes = []
        for b in self._binned_spec_with_disp_list:
            fluxes.append(b._evaluate_model(precalc_fluxes=precalc_fluxes))
        fluxes = np.array(fluxes)
        return fluxes

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
        models = np.empty((len(self._binned_spec_with_disp_list),))
        for i, b in enumerate(self._binned_spec_with_disp_list):
            models[i] = b.get_model(precalc_fluxes)
        return models

    def get_log_like(self):
        pass

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
