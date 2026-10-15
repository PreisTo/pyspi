import logging
import os

from astromodels.functions.priors import Cosine_Prior, Uniform_prior
from threeML import DispersionSpectrumLike
from threeML.utils.spectrum.binned_spectrum import BinnedSpectrumWithDispersion

from pyspi.utils.cluster_fit_core import extract_pointing_info
from pyspi.utils.function_utils import find_response_version
from pyspi.utils.response.spi_drm import SPIDRM
from pyspi.utils.response.spi_response import ResponseRMFGenerator
from pyspi.utils.rsp_bases import get_base_rsp

log = logging.getLogger(__name__)


class SPIPointingPlugin(DispersionSpectrumLike):
    def __init__(
        self,
        name,
        data,
        background=None,
        free_position=False,
        verbose=True,
        use_cache=False,
        **kwargs,
    ):

        assert isinstance(data.response, SPIDRM), "Response is not a SPI response"
        super().__init__(name, data, background, verbose, **kwargs)

    def set_model(self, likelihoodModel):
        """
        Set the model and free the location parameters


        :param likelihoodModel:
        :return: None
        """

        # set the standard likelihood model

        super().set_model(likelihoodModel)

        if self._free_position:
            log.info(f"Freeing the position of {self.name} and setting priors")

            for key in self._like_model.point_sources:
                self._like_model.point_sources[key].position.ra.free = True
                self._like_model.point_sources[key].position.dec.free = True

                self._like_model.point_sources[key].position.ra.prior = Uniform_prior(
                    lower_bound=0.0, upper_bound=360
                )
                self._like_model.point_sources[key].position.dec.prior = Cosine_Prior(
                    lower_bound=-90.0, upper_bound=90
                )

                ra = self._like_model.point_sources[key].position.ra.value
                dec = self._like_model.point_sources[key].position.dec.value

        else:
            for key in self._like_model.point_sources:
                ra = self._like_model.point_sources[key].position.ra.value
                dec = self._like_model.point_sources[key].position.dec.value

        self._response.set_location(ra, dec, cache=self._use_cache)

    def get_log_like(self):
        raise RuntimeError("You must not use singular SPI pointings")

    def get_model(self, precalc_fluxes=None):

        if self._free_position:
            ra, dec = self._like_model.get_point_source_position(0)

            self._response.set_location(ra, dec, cache=self._use_cache)

        return super().get_model(precalc_fluxes)

    def rebin_on_source(self, *args, **kwargs):
        raise NotImplementedError(
            "This function can not be called on singular SPI plugins - use the set"
        )

    def rebin_on_background(self, *args, **kwargs):
        raise NotImplementedError(
            "This function can not be called on singular SPI plugins - use the set"
        )

    @classmethod
    def from_pointing_number(cls, pointing_number, det, energy_bins, mc_energies):
        # TODO: this only works for singles currently
        path = os.path.join(
            os.environ.get("PYSPI", None),
            "pointing_data",
            pointing_number,
        )

        time_start, time_elapsed, energy_bins, counts_f = extract_pointing_info(
            path, pointing_number, energy_bins=energy_bins
        )
        version = find_response_version(time_start)
        rsp_base = get_base_rsp(version)
        rsp_gen = ResponseRMFGenerator.from_time(
            time_start, det, energy_bins, mc_energies, rsp_base
        )
        rsp = SPIDRM(rsp_gen, 0, 0)

        bsp = BinnedSpectrumWithDispersion(
            counts=counts_f[det],
            exposure=time_elapsed[det],
            response=rsp,
            is_poisson=True,
            mission="INTEGRAL",
            instrument=f"SPI{det}",
            # tstart=datetime.strptime(time_start, "%y%m%d %H%M%S"),
        )
        return cls(name=f"SPIID{pointing_number}det{det}", data=bsp)

    @classmethod
    def from_data(
        cls,
        time_start,
        time_elapsed,
        counts,
        energy_bins,
        mc_energies,
        pointing_number,
        det,
    ):
        version = find_response_version(time_start)
        rsp_base = get_base_rsp(version)
        rsp_gen = ResponseRMFGenerator.from_time(
            time_start, det, energy_bins, mc_energies, rsp_base
        )
        rsp = SPIDRM(rsp_gen, 0, 0)

        bsp = BinnedSpectrumWithDispersion(
            counts=counts,
            exposure=time_elapsed,
            response=rsp,
            is_poisson=True,
            mission="INTEGRAL",
            instrument=f"SPI{det}",
            # tstart=datetime.strptime(time_start, "%y%m%d %H%M%S"),
        )
        return cls(name=f"SPIID{pointing_number}det{det}", data=bsp)
