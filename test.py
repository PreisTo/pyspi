#! /usr/bin/env python3

from datetime import datetime

import numpy as np
from astromodels import Log_uniform_prior, Model, PointSource, Powerlaw, Uniform_prior
from astropy.coordinates import SkyCoord
from astropy.time import Time
from threeML import BayesianAnalysis, DataList, plot_spectra

from pyspi.clustering.pointing import PointingClusters
from pyspi.SPIClusterLike import SPIClusterLike
from pyspi.utils.query import SelectData


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


if __name__ == "__main__":
    target = SkyCoord.from_name("XTE J1817-330")
    sd = SelectData(target)
    start = datetime(2006, 2, 15)
    stop = datetime(2006, 2, 18)
    sd.filter(
        time_range=(Time(start).mjd, Time(stop).mjd),
        min_good_spi=800,
    )
    print(sd.table)
    pc = PointingClusters(sd, cluster_size_range=(3, 3))
    print(pc.pointings)
    sl = SPIClusterLike(
        "test",
        pc.pointings,
        energy_range=(20, 1000),
        emod=np.geomspace(20, 1000, 500),
        binning_func=no_rebinning,
        source_spectrum_powerlaw_binning=False,
    )

    pl = Powerlaw()
    ps = PointSource("test", ra=target.ra.deg, dec=target.dec.deg, spectral_shape=pl)
    model = Model(ps)
    pl.K.prior = Log_uniform_prior(lower_bound=1e-3, upper_bound=1e3)
    pl.index.prior = Uniform_prior(lower_bound=-5, upper_bound=2)
    dl = DataList(sl)

    ba = BayesianAnalysis(model, dl)
    ba.set_sampler("nautilus")
    ba.sampler.setup(pool=10)
    ba.sample()
    print(ba.results)
    fig = plot_spectra(ba.results)
    fig.savefig("test.png")

    fig = ba.results.corner_plot()
    fig.savefig("corner.png")
