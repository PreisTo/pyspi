from typing import Optional

import numpy as np
from astromodels.core.model import Model
from astromodels.functions.priors import Cosine_Prior, Uniform_prior
from threeML.plugin_prototype import PluginPrototype
from pyspi.utils.likelihood_helpers import logLcore
from pyspi.utils.query import SelectData
from pyspi.clustering.pointing import PointingClusters

from pyspi.utils.response.spi_drm import SPIDRM
import logging

log = logging.getLogger(__name__)


class SPIConstantLike(PluginPrototype):
    """
    Plugin for the data of SPI, based on PySPI
    """

    def __init__(
        self,
        name: str,
        pointings: list,
        energy_range: np.array = np.geomspace(20, 2000, 40),
    ):
        """
        Initialize the SPIConstantLike plugin.

        :param name: Name of the plugin
        :param spec_binned: Binned spectrum
        :param pointings: List of number of pointings
        :param dets: List of number of detectors
        :param resp_mats: List of response matrices
        :param num_sources: Number of sources
        :param t_elapsed: List of elapsed times
        :param counts: List of counts
        :param drm: Optional SPIDRM object for response matrix handling
        """
        super().__init__()

    def get_log_like(self, precalc_fluxed: Optional[np.ndarray] = None) -> float:
        """
        Calculate the log-likelihood for the given model.

        :param model: The model to evaluate
        :return: The log-likelihood value
        """
        spec_binned = self.get_binned_spectrum()
        logL = logLcore(
            spec_binned,
            self._pointings,
            self._dets,
            self.resp_mats,
            self.num_sources,
            self.t_elapsed,
            self.counts,
        )
        return logL

    def set_model(self, likelihood_model):
        pass

    @classmethod
    def from_query(
        cls,
        query: SelectData,
        min_angle_dif=1,
        max_angle_dif=10,
        max_time_dif=0.2,
        min_time_elapsed=300,
        cluster_size_range=(2, 2),
        random_angle_dif_range=None,
    ):
        """
        Create an instance of SPIConstantLike from a SelectData query.

        :param query: SelectData object containing the query results
        :return: An instance of SPIConstantLike
        """

        cluster = PointingClusters(
            query_table=query.table,
            min_angle_dif=min_angle_dif,
            max_angle_dif=max_angle_dif,
            max_time_dif=max_angle_dif,
            min_time_elapsed=min_time_elapsed,
            cluster_size_range=cluster_size_range,
            random_angle_dif_range=random_angle_dif_range,
            center_ra=query.position.ra.deg,
            center_dec=query.position.dec.deg,
        )
        return cls()
