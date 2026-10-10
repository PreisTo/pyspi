from datetime import datetime

import numpy as np
from threeML import DataList, BayesianAnalysis
from astromodels.functions import Powerlaw, Uniform_prior, Log_uniform_prior
from astromodels.sources import PointSource
from astromodels.core.model import Model
from astropy.coordinates import SkyCoord
from astropy.time import Time

from pyspi.clustering.pointing import PointingClusters
from pyspi.SPIClusterLike import SPIClusterLike
from pyspi.utils.query import SelectData
import matplotlib.pyplot as plt


def no_rebinning(bins, counts, energy_range):
    if energy_range[0]:
        for i, e in enumerate(bins):
            if e >= energy_range[0]:
                bins = bins[i:]
                counts = counts[:, i:]
                break
    if energy_range[1]:
        for i, e in enumerate(bins):
            if e > energy_range[1]:
                bins = bins[:i]
                counts = counts[:, : i - 1]
                assert i > 1, "Max Energy is too low"
                break

    return bins, counts


target_position = SkyCoord.from_name("Crab")

pl = Powerlaw()
ps = PointSource(
    "crab", ra=target_position.ra.deg, dec=target_position.dec.deg, spectral_shape=pl
)
model = Model(ps)

pl.K.prior = Log_uniform_prior(lower_bound=1e-4, upper_bound=1e3)
pl.index.prior = Uniform_prior(lower_bound=-4, upper_bound=-1)
sd = SelectData(target_position)
start = datetime(2003, 1, 1)
stop = datetime(2003, 12, 31)
sd.filter(
    time_range=(Time(start).mjd, Time(stop).mjd),
    min_good_spi=800,
)
sd.table = sd.table[:9]
print(sd.table)
pc = PointingClusters(sd, cluster_size_range=(3, 3))
print(pc.pointings)
sl = SPIClusterLike(
    "spi_data",
    pc.pointings,
    energy_range=(20, 1000),
    emod=np.geomspace(20, 1000, 500),
    binning_func=no_rebinning,
    source_spectrum_powerlaw_binning=False,
)

dl = DataList(sl)
ba = BayesianAnalysis(model, dl)
ba.set_sampler("multinest")
ba.sampler.setup(resume=False)
ba.sample()
ba.results.write_to("test.fits", overwrite=True)
fig = ba.results.corner_plot()
fig.savefig("test.png")
