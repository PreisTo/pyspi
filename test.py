#! /usr/bin/env python3

from pyspi.utils.query import SelectData
from pyspi.clustering.pointing import PointingClusters
from pyspi.SPIClusterLike import SPIClusterLike
from astropy.coordinates import SkyCoord
from datetime import datetime, timedelta
import numpy as np
from astropy.time import Time


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
    sd = SelectData(SkyCoord.from_name("G 21.5-0.9"))
    start = datetime(2008, 1, 1)
    sd.filter(
        time_range=(
            float(Time(start).mjd),
            float(Time(start + timedelta(days=31 * 2)).mjd),
        )
    )
    pc = PointingClusters(sd, cluster_size_range=(2, 2))
    print(pc.pointings)
    sl = SPIClusterLike(
        "test",
        pc.pointings,
        energy_range=(20, 1000),
        emod=np.geomspace(20, 1000, 500),
        binning_func=no_rebinning,
        source_spectrum_powerlaw_binning=False,
    )
