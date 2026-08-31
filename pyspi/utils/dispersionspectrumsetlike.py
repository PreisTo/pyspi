import logging
import numpy as np
from typing import Optional

from threeML.plugins.DipersionSpectrumLike import DipersionSpectrumLike

log = logging.getLogger(__name__)


class DipersionSpectrumSetLike(DipersionSpectrumLike):
    def __init__(
        self,
        name,
        binned_spec_with_disp_list,
        **kwargs,
    ):
        self._binned_spec_with_disp_list = binned_spec_with_disp_list
        rsps = []
        for b in self._binned_spec_with_disp_list:
            rsps.append(b.response)
        self._responses = rsps

    def set_model(self, likelihoodmodel):
        for bs in self._binned_spec_with_disp_list:
            bs.set_model(likelihoodmodel)

    def _evaluate_model(self, precalc_fluxes: Optional[np.ndarray] = None):
        fluxes = []
        for r in self._respo:
            fluxes.append(r.convolve(precalc_fluxes=precalc_fluxes))
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
            for r in self._responses:
                r.set_function(self._integral_flux)

    def display_rsp(self):
        figs = []
        for r in self.responses:
            figs.append(r.plot_matrix())
        return figs

    @property
    def responses(self):
        return self._responses

    def get_log_like(self):
        pass
