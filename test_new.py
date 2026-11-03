import time
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astromodels import (
    Powerlaw,
    PointSource,
    Model,
    Log_uniform_prior,
    Uniform_prior,
)
from threeML import (
    DataList,
    BayesianAnalysis,
    display_spectrum_model_counts,
    plot_spectra,
)
from pyspi.utils.dispersionspectrumsetlike import DispersionSpectrumSetLike
from pyspi.utils.query import SelectData
from pyspi.clustering.pointing import PointingClusters
from datetime import datetime
from astropy.time import Time

import cProfile, pstats, io
from pstats import SortKey
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

target = SkyCoord.from_name("Crab")
sd = SelectData(target)
start = datetime(2003, 1, 1)
stop = datetime(2003, 3, 31)


sd.filter(
    time_range=(Time(start).mjd, Time(stop).mjd),
    min_good_spi=1000,
    radius=10 * u.deg,
    scw_id_range=(43 * 1e8, 46 * 1e8),  # all scws within rev 43 - 45
)
sd.table = sd.table
pc = PointingClusters(
    sd, cluster_size_range=(3, 6), cluster_size_preference_threshold=(2.0, 2.0, 2.0)
)
if rank == 0:
    print(len(pc.cluster_list), flush=True)
    pc.get_cluster_infos()


data = []
time.sleep(rank * 5)
cluster_list = pc.cluster_list
for c in range(len(cluster_list)):
    dsp = DispersionSpectrumSetLike.from_pointing_list(
        str(cluster_list[c]),
        cluster_list[c],
        np.geomspace(20, 600, 41),
        np.geomspace(10, 1000, 500),
        pointing_pos=[target.ra.deg, target.dec.deg],
    )
    data.append(dsp)
pl = Powerlaw()
ps = PointSource("test", ra=target.ra.deg, dec=target.dec.deg, spectral_shape=pl)
model = Model(ps)
pl.piv.value = 100

dsp.set_model(model)

pl.index.set_uninformative_prior(Uniform_prior)
pl.K.set_uninformative_prior(Log_uniform_prior)


ba = BayesianAnalysis(model, DataList(*data))
ba.set_sampler("multinest")
ba.sampler.setup()
comm.Barrier()
pr = cProfile.Profile()
pr.enable()
ba.sample()
pr.disable()
if rank == 0:
    print(ba.results)
    fig = ba.results.corner_plot()
    fig.savefig("test_corner.png")
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(30)
    print(s.getvalue())

    print(
        ba.results.get_flux(
            ene_min=50 * u.keV, ene_max=100 * u.keV, flux_unit="cm-2 s-1"
        )
    )
    fig = display_spectrum_model_counts(ba)
    fig.savefig("test_counts.png")
    fig = plot_spectra(ba.results)
    fig.savefig("test_spectrum.png")
