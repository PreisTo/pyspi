from threeML.plugins.DipersionSpectrumLike import DipersionSpectrumLike

import logging

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
        log.debug(f"model set for {self._name}")

        self._like_model = likelihoodmodel

        if self._source_name is not None:
            if self._source_name not in self._like_model.sources.keys():
                log.error(
                    "Source %s is not contained in "
                    "the likelihood model" % self._source_name
                )

                raise RuntimeError()
            if self._source_name in self._like_model.extended_sources.keys():
                log.error(
                    "SpectrumLike plugins do not support extended sources."
                    "You specified an extended source via the assign_to_source "
                    f"function - source name is {self._source_name}."
                )

                raise RuntimeError()

        else:
            if self._like_model.get_number_of_extended_sources() > 0:
                log.error(
                    "SpectrumLike plugins do not support extended sources."
                    "In case those are not relevant for that plugin use "
                    "assign_to_source to assign a PointSource."
                )

                raise RuntimeError()

        if hasattr(self, "_get_diff_flux_and_integral"):  # check if we have that
            # function as I am trying to get rid of it for MPI reasons
            # Get the differential flux function, and the integral function, with no
            # dispersion, we simply integrate the model over the bins

            differential_flux, integral = self._get_diff_flux_and_integral(
                self._like_model, integrate_method=self._model_integrate_method
            )

            log.debug(f"{self._name} passing integral flux function to RSP")

            self._integral_flux = integral
        else:
            pass

        # pass to the response matrix
        for r in self._responses:
            r.set_function(self._integral_flux)

    def _evaluate_model(self):
        pass

    def set_model_integrate_method(self):
        pass

    def display_rsp(self):
        figs = []
        for r in self.responses:
            figs.append(r.plot_matrix())
        return figs

    @property
    def responses(self):
        self._responses

    def get_log_like(self):
        pass
